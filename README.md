# SAMI project 

Automatic System to Measure Speech Intelligibility (SAMI in French).
It is a blind unsupervised system designed to be close to the intelligibility of production and to the severity index of speech disorder measures.
Launching the `Pase+_results.py` file launch the experiments over the [C2SI corpus](https://www.irit.fr/publis/SAMOVA/JOU/LRE_2019-soumission.pdf).
All resulting experiments on this corpus can be found in `results` folder.
There you will find 3 subfolders (`all`, `controls` and `files`) corresponding to normalization using all features (`all`), using only controls features (`controls`) and using only the processed file (`file`).
All computations to the KL divergence are done in `tools/pase_score.py` module, the rest is in `Pase+_results.py`.
This work is under submission process for the ICASSP2021 conference.
The original parameters learned by [Ravanelli et al. 2020](https://arxiv.org/abs/2001.09239) over 50h of the LibriSpeech dataset for the Pase+ model is under the `trained_model/PASE+_parameters.ckpt` file.


## Dependencies

This repository depends on the [audio_loader](https://github.com/vroger11/audio_loader) project and the [PASE+ model](https://github.com/santi-pdp/pase).
The audio_loader and PASE+ projects have to be in the PYTHONPATH of the user.
All other requirements are listed in the `pase+_environment.yml` file.
This file can be used to create a `pase+` environment with anaconda.

# Contacts

* Vincent Roger - Vincent.Roger@irit.fr
* Jérôme Farinas - Jerome.Farinas@irit.fr
* Virginie Woisard - woisard.v@chu-toulouse.fr
* Julien Pinquier - Julien.Pinquier@irit.fr
