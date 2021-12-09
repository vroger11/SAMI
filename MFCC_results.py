"""Module to do MFCC experiments."""
import argparse
import sys
import librosa

from bokeh.models import transforms
from os.path import join
from pathlib import Path
from math import e as euler_number
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem

import torch
import numpy as np
import pickle

from scipy.stats import spearmanr

from tools.pase_score import pase_score
from tools.c2si_data import get_c2si_dataloader, C2SI_FOLDERPATH
from tools.plots import plot_results


CUDA0 = torch.device('cuda:0')
SCORES = ["std_score", "mean_score", "median_score", "min_score", "max_score"]
METRICS = ["severity", "intel"]

def details_by_filepath(kl_score, c2si_gt):
    res = {}
    for fp in kl_score:
        res[fp] = {}
        res[fp]["kl_d"] = kl_score[fp]
        selected_row = c2si_gt.df_selected[c2si_gt.df_selected["relative path"] == c2si_gt.get_id(fp)]
        res[fp]["intel"] = selected_row["intel"].values[0]
        res[fp]["severity"] = selected_row["sev"].values[0]
        res[fp]["group"] = selected_row["group"].values[0]
        res[fp]["sex"] = selected_row["sex"].values[0]
        res[fp]["age"] = selected_row["age"].values[0]

    return res

def prepare_source(res_all):
    """Creating dictionnary for bokeh plotting source."""
    patients = []
    res_plot = {"max_score": [], "min_score": [], "median_score": [], "mean_score": [], "std_score": [], "intel": [], "severity": [], "group": [], "filepath": [], "sex": []}
    for fn in res_all:
        # rewrite the exponential value because it appears to be not stable on gpu
        res_plot["max_score"].append(float(torch.pow(euler_number, res_all[fn]["kl_d"].min()).detach().cpu().numpy()))
        res_plot["min_score"].append(float(torch.pow(euler_number, res_all[fn]["kl_d"].max()).detach().cpu().numpy()))
        res_plot["median_score"].append(float(torch.pow(euler_number, res_all[fn]["kl_d"].median()).detach().cpu().numpy()))
        res_plot["mean_score"].append(float(torch.pow(euler_number, res_all[fn]["kl_d"].mean()).detach().cpu().numpy()))
        res_plot["std_score"].append(float(torch.pow(euler_number, res_all[fn]["kl_d"].std()).detach().cpu().numpy()))

        res_plot["group"].append("red" if res_all[fn]["group"] == 1 else "blue")
        patients.append(res_all[fn]["group"] == 1)
        res_plot["filepath"].append(fn)
        res_plot["severity"].append(res_all[fn]["severity"])
        res_plot["intel"].append(res_all[fn]["intel"])
        sex = "man" if res_all[fn]["sex"] == 1 else "woman"
        if res_all[fn]["sex"] == 1:
            sex = "man"
        elif res_all[fn]["sex"] == 2:
            sex = "woman"
        else:
            sex = "unkown"

        res_plot["sex"].append(sex)

    return res_plot, patients


def main(folder_out="c2si_results", seg_size=16000, folder_in=C2SI_FOLDERPATH, transform=False):
    """
    Parameters
    ----------
    folder_out: str, optional
        Folder containing the results of the experiment.

    seg_size: int, optional
        Number of samples to take, here for the C2SI corpus 16000 samples correspond to 1s

    """
    # load model
    def mfcc_model(batch):
        res = []
        for segment in batch:
            res.append(librosa.feature.mfcc(y=segment.flatten().cpu().numpy(), sr=16000))

        res = torch.tensor(res)
        return res


    # load data
    all_dataloader, c2si_gt = get_c2si_dataloader("all", seg_size=seg_size, overlap=0., normalize_audio=True, vad=[0.2, 0.5, 5], c2si_folderpath=folder_in)

    # Compute results
    feature_preparations = [None, "l1", "l2", "max", "inf", "zscore"]
    feature_prep_strats = ["all", "file", "controls"]
    best_result = None
    for feature_prep_strat in feature_prep_strats:
        folder_result = join(folder_out, feature_prep_strat)
        for feature_preparation in feature_preparations:
            kl_d, _ = pase_score(mfcc_model, all_dataloader, c2si_gt,
                                 norm=feature_preparation, norm_approach=feature_prep_strat,
                                 device=CUDA0)
            if feature_preparation is None:
                feature_preparation = "only_absolute"

            res_all = details_by_filepath(kl_d, c2si_gt)

            res_plot, patients = prepare_source(res_all)
            if feature_prep_strat == "controls" and feature_preparation == "max":
                best_result = res_plot

            # Bokeh plots
            metrics = ["intel", "severity"]
            for y_axis in metrics:
                for score in SCORES:
                    plot_results("MFCC", res_plot, seg_size, feature_preparation,
                                 score_version=score.replace("_score", ""), y_axis=y_axis,
                                 folder_out=folder_result)

            result_file = open(join(folder_result, f"{feature_preparation}.txt"), "w")
            result_file.write(f"{feature_preparation} results\n\nAll participants results\n")
            # Correlations for all
            for metric in metrics:
                for score in SCORES:
                    result_file.write(f"Correlation {score}/{metric}:\n")
                    result_file.write("spearmanr " + str(round(spearmanr(res_plot[score], res_plot[metric])[0], 4)) + "\n")

            result_file.write("\nPatients only results:\n")

            # Correlations on patients only
            for metric in metrics:
                # select on patients results
                patient_metric = np.array(res_plot[metric])[patients]
                for score in SCORES:
                    result_file.write(f"Correlation {score}/{metric}:\n")
                    result_file.write("spearmanr " + str(round(spearmanr(np.array(res_plot[score])[patients], patient_metric)[0], 4)) + "\n")

            result_file.write("\n")
            result_file.close()

    if transform:
        print("Save tranformation of the score into severity score")
        data = best_result["mean_score"]
        data = np.array(data).reshape((-1, 1))
        target = np.array(best_result["severity"])

        nystroem_regression = make_pipeline(
            Nystroem(n_components=5), LinearRegression(),
        )
        nystroem_regression.fit(data, target)
        target_predicted = np.clip(nystroem_regression.predict(data), 0, 10)
        mse = mean_squared_error(target, target_predicted)
        print(f"Achieved MSE: {mse}")

        # save the classifier
        with open('regression_model.pkl', 'wb') as fid:
            pickle.dump(nystroem_regression, fid)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("folder_out", type=str, help="Folder where the results will be put.")
    PARSER.add_argument("-s", "--seg_size", type=int, default=16000,
                        help="Size of segment used (in number of samples).")
    PARSER.add_argument("-d", "--data", type=str,
                        default=C2SI_FOLDERPATH,
                        help="C2SI folder path.")
    PARSER.add_argument("-t", "--transformation",
                    action="store_true",
                    help="Compute transformation function of model score into severity score.")
    ARGS = PARSER.parse_args()

    main(ARGS.folder_out, ARGS.seg_size, ARGS.data, ARGS.transformation)
