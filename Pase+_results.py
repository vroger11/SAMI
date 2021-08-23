"""Module to do Pase+ experiments."""
import argparse
import sys
from os.path import join
from pathlib import Path
from math import e as euler_number

import torch
import numpy as np

from scipy.stats import spearmanr

from tools.pase_score import pase_score
from tools.c2si_data import get_c2si_dataloader, C2SI_FOLDERPATH
from tools.plots import plot_results

PASE_FOLDER = join(str(Path.home()), 'git/pase+')
sys.path.append(PASE_FOLDER)
from pase.models.frontend import wf_builder


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


# Load model
def load_pase_plus(pase_folder=PASE_FOLDER, parameters='trained_model/PASE+_parameters.ckpt'):
    pase = wf_builder(join(PASE_FOLDER, 'cfg/frontend/PASE+.cfg'))
    pase.eval()
    pase.load_pretrained(parameters, load_last=True, verbose=True)
    return pase.to(CUDA0)


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


def main(folder_out="c2si_results", seg_size=16000, parameters='trained_model/PASE+_parameters.ckpt', folder_in=C2SI_FOLDERPATH):
    """
    Parameters
    ----------
    folder_out: str, optional
        Folder containing the results of the experiment.

    seg_size: int, optional
        Number of samples to take, here for the C2SI corpus 16000 samples correspond to 1s

    parameters: str, optional
        Filepath to the pase+ parameters

    """
    # load model
    pase = load_pase_plus(parameters=parameters)
    # load data
    all_dataloader, c2si_gt = get_c2si_dataloader("all", seg_size=seg_size, overlap=0., normalize_audio=True, vad=[0.2, 0.5, 5], c2si_folderpath=folder_in)

    # Compute results
    feature_preparations = [None, "l1", "l2", "max", "inf", "zscore"]
    feature_prep_strats = ["all", "file", "controls"]
    for feature_prep_strat in feature_prep_strats:
        folder_result = join(folder_out, feature_prep_strat)
        for feature_preparation in feature_preparations:
            kl_d, _ = pase_score(pase, all_dataloader, c2si_gt,
                                 norm=feature_preparation, norm_approach=feature_prep_strat,
                                 device=CUDA0)
            if feature_preparation is None:
                feature_preparation = "only_absolute"

            res_all = details_by_filepath(kl_d, c2si_gt)

            res_plot, patients = prepare_source(res_all)
            # Bokeh plots
            metrics = ["intel", "severity"]
            for y_axis in metrics:
                for score in SCORES:
                    plot_results("Pase+", res_plot, seg_size, feature_preparation,
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

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("folder_out", type=str, help="Folder where the results will be put.")
    PARSER.add_argument("-s", "--seg_size", type=int, default=16000,
                        help="Size of segment used (in number of samples).")
    PARSER.add_argument("-p", "--pase_plus_parameters", type=str,
                        default='trained_model/PASE+_parameters.ckpt',
                        help="Filepath to the parameters to use.")
    PARSER.add_argument("-d", "--data", type=str,
                        default=C2SI_FOLDERPATH,
                        help="C2SI folder path.")
    ARGS = PARSER.parse_args()

    main(ARGS.folder_out, ARGS.seg_size, ARGS.pase_plus_parameters, ARGS.data)
