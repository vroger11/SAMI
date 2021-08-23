import itertools
import torch
import torch.nn.functional as F
import numpy as np
import pickle

from tools.functional import PreProcess


def pase_score(model, dataloader, al_gt, norm="l1", norm_approach="all", device="cpu", save_controls_parameters=True):
    """Inspired from the inception score.

    Parameters
    ----------
    model: nn.Module
        model used to get the encodings.

    dataloader:
        Dataloader construct with the audio_loader project.

    al_gt: audio_loader.ground_truth.C2SI
        C2SI ground truth like from audio_loader project.

    norm_approach: str, optional
        Can be either "all", "file" or "controls".
    """
    predictions = []
    filepaths = []

    with torch.no_grad():
        # get predictions and filepaths of model
        for batch in dataloader:
            shape = batch[0][0].shape
            batch_samples = batch[0][0].reshape(shape[0], shape[1], shape[2]*shape[3])
            filepaths.append(batch[1])
            encoding = model(batch_samples.to(device))
            predictions.append(encoding)

        p_yx_all = torch.cat(predictions, 0)
        p_yx_all = p_yx_all.transpose(1, 2)
        # to have more precision as we deal with small numbers
        p_yx_all = p_yx_all.to(device=device, dtype=torch.double)
        filepaths = list(itertools.chain(*filepaths))

        featureProcess = PreProcess(norm)
        # fit parameters for strategy compatible
        if norm_approach == "all":
            featureProcess.fit(p_yx_all)
        elif norm_approach == "controls":
            p_yx_controls = []
            fp_unique = np.unique(filepaths)
            for fp in fp_unique:
                p_yx = torch.cat([p_yx_all[i] for i in range(len(filepaths)) if filepaths[i] == fp])
                selected_row = al_gt.df_selected[al_gt.df_selected["relative path"] == al_gt.get_id(fp)]
                if selected_row["group"].values[0] == 2:
                    p_yx_controls.append(p_yx)

            featureProcess.fit(torch.cat(p_yx_controls))

            if save_controls_parameters:
                fileobject = open(f"controls_featureProcess_{norm}.pickle", 'wb')
                pickle.dump(featureProcess, fileobject)
                fileobject.close()

        # group by participants (one audio recording file is viewed as a participatant)
        # and compute KL divergence
        p_yx_participants = {}
        KL_d = {}
        fp_unique = np.unique(filepaths)
        for fp in fp_unique:
            # Compute kl_divergence over all examples of a file
            p_yx = torch.cat([p_yx_all[i] for i in range(len(filepaths)) if filepaths[i] == fp])

            if norm_approach == "file":
                featureProcess.fit(p_yx)

            p_yx = featureProcess.transform(p_yx)
            # extreme negative values should have the same impact as extreme positive values
            p_yx = p_yx.abs()
            p_yx = F.softmax(p_yx, 1)

            p_yx_participants[fp] = p_yx # copy to analyse results
            p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
            KL_d[fp] = p_yx * (torch.log(p_yx) - torch.log(p_y))

        return KL_d, p_yx_participants
