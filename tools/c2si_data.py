"""Module using Audio Loader project to load the C2SI corpus."""
from os.path import join
from pathlib import Path

from audio_loader.features.raw_audio import WindowedAudio
from audio_loader.ground_truth.c2si import C2SI
from audio_loader.samplers.windowed import WindowedSampler
from audio_loader.dl_frontends.pytorch.fill_ram import get_pytorch_dataloader_fill_ram
from audio_loader.activity_detection.simple_VAD import Simple


C2SI_FOLDERPATH = join(Path.home(), "data/parolotheque_v2")

def get_c2si_dataloader(group="all", seg_size=4000, overlap=0., c2si_folderpath=C2SI_FOLDERPATH, normalize_audio=True, vad=None):
    """Prepare dataloader corresponding to C2SI data.

    Parameters
    ----------
    group: str, optional
        Can be either "all", "controls" or "patients" participants.

    seg_size: int, optional
        Number of samples for an example.

    overlap: float, optional
        Values should be from 0. to 1. excluded. This is the percentage of overlap applied over the
        audio signals.

    c2si_folderpath: str, optional


    normalize_audio: bool, optional
        If True, normalize each audio signal by the maximal absolute value of each signal.

    vad: list or None, optional
        Voice Activity Detection parameters based on the simple system in audio_loader project.
        If listed, the parameters correspond to:
            [energy_threshold, spectral_flatness_threshold, smooth].
        For more details  audio_loader.activity_detection.simple_VAD.Simple class.

    Returns
    -------
    dataloader: torch.DataLoader
        Dataloader corresponding to the parameters given.
    """
    win_size, hop_size = 1024, 1024

    c2si_gt = C2SI(c2si_folderpath,
                   "16k", "BASE-Version-2020-02-14.xlsx",
                   sets=None, severity_score=True,
                   targeted_tasks=['L'], group=group)

    if vad is not None:
        vad = Simple(win_size, hop_size, 16000,
                     energy_threshold=vad[0], spectral_flatness_threshold=vad[1], smooth=vad[2])
        padding = True
    else:
        padding = False

    raw_feature_processor = WindowedAudio(win_size, hop_size, 16000,
                                          normalize=normalize_audio, padding=padding)
    raw_sampler = WindowedSampler([raw_feature_processor], c2si_gt, seg_size,
                                  overlap=overlap, output_filepath=True, activity_detection=vad)

    return get_pytorch_dataloader_fill_ram(raw_sampler, 16, "test"), c2si_gt
