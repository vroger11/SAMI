"""Module to do Pase+ experiments."""
import argparse
import sys
import pickle
from os.path import join
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F

from audio_loader.features.raw_audio import WindowedAudio
from audio_loader.samplers.windowed_segments import WindowedSegmentSampler
from audio_loader.dl_frontends.pytorch.fill_ram import get_dataloader_fixed_size
from audio_loader.activity_detection.simple_VAD import Simple


PASE_FOLDER = join(str(Path.home()), 'git/pase+')
sys.path.append(PASE_FOLDER)
from pase.models.frontend import wf_builder


CUDA0 = torch.device('cuda:0')
SCORES = ["std_score", "mean_score", "median_score", "min_score", "max_score"]



# Load model
def load_pase_plus(pase_folder=PASE_FOLDER, parameters='trained_model/PASE+_parameters.ckpt'):
    pase = wf_builder(join(PASE_FOLDER, 'cfg/frontend/PASE+.cfg'))
    pase.eval()
    pase.load_pretrained(parameters, load_last=True, verbose=True)
    return pase.to(CUDA0)

def load_control_normalizer(filename):
    fsaved = open(filename, "rb")
    return pickle.load(fsaved)


def extract_data(filename, seg_size=16000):
    """Extrract the data to forward from one file.

    """
    win_size, hop_size, overlap = 1024, 1024, 0.
    normalize_audio = True

    vad = [0.2, 0.5, 5]
    vad = Simple(win_size, hop_size, seg_size,
                 energy_threshold=vad[0], spectral_flatness_threshold=vad[1], smooth=vad[2])
    padding = True

    raw_feature_processor = WindowedAudio(win_size, hop_size, seg_size,
                                          normalize=normalize_audio, padding=padding)
    raw_sampler = WindowedSegmentSampler([raw_feature_processor], None, seg_size,
                                         overlap=overlap, output_filepath=False,
                                         supervised=False,
                                         activity_detection=vad)

    return raw_sampler
    #return get_dataloader_fixed_size(raw_sampler, 16, "test")


def compute_pase_score(model, filepath, featureProcess, sampler, device="cpu"):
    """Inspired from the inception score.

    Parameters
    ----------
    model: nn.Module
        model used to get the encodings.

    al_gt: audio_loader.ground_truth.C2SI
        C2SI ground truth like from audio_loader project.
    """
    predictions = []

    with torch.no_grad():
        # get predictions and filepaths of model
        batch_samples = [x[0] for x in sampler.get_samples_from(selected_set=[filepath])]
        batch = torch.tensor(batch_samples, device='cuda')
        shape = batch.shape
        batch_samples = batch.reshape(shape[0], 1, shape[1]*shape[2])
        encoding = model(batch_samples.to(device))
        predictions.append(encoding)

        p_yx_all = encoding
        p_yx_all = p_yx_all.transpose(1, 2)
        # to have more precision as we deal with small numbers
        p_yx = p_yx_all.to(device=device, dtype=torch.double)
        p_yx = p_yx.reshape(p_yx.size(0)*p_yx.size(1), p_yx.size(-1))


        # group by participants (one audio recording file is viewed as a participatant)
        # Compute kl_divergence over all examples of a file

        p_yx = featureProcess.transform(p_yx)
        # extreme negative values should have the same impact as extreme positive values
        p_yx = p_yx.abs()
        p_yx = F.softmax(p_yx, 1)

        p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
        KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))

        return KL_d


def main(filepath, seg_size=16000, parameters='trained_model/PASE+_parameters.ckpt'):
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
    sampler = extract_data(filepath)

    # load precomputed controls
    feature_process = load_control_normalizer("./controls_featureProcess_max.pickle")
    # Compute kl
    kl_d = compute_pase_score(pase, filepath, feature_process, sampler, device=CUDA0)

    # compute final score
    score = kl_d.mean().cpu().numpy()

    print(f"score: {score}")

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-f", "--filepath", type=str,
                        help="File path to test.")
    PARSER.add_argument("-s", "--seg_size", type=int, default=16000,
                        help="Size of segment used (in number of samples).")
    PARSER.add_argument("-p", "--pase_plus_parameters", type=str,
                        default='trained_model/PASE+_parameters.ckpt',
                        help="Filepath to the parameters to use.")
    ARGS = PARSER.parse_args()

    main(ARGS.filepath, ARGS.seg_size, ARGS.pase_plus_parameters)
