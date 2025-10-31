Speech rhythm analysis tools

# Code purpose and functionalities

This repository contains code associated with the Master's project titled "Enhancing duration prediction in deep learning TTS models from a psychoacoustic perspective of speech rhythm". The ``vowel_beat_detector`` submodule provides rhythmic feature extraction tools (check the submodule's README.md for more details). The source code at root provides deep learning resources to process these features and perform feature learning and classification tasks.

# Quick start

1. Clone this repository and set up the environment
    ```
    cd <path to cloned repo>
    conda env create -f environment.yml
    conda activate rtm
    ```

2. Download datasets and extract features
    ```
    python vowel_beat_detector/src/main.py \
    -in <dataset_path> \
    -out data/<dataset_name> \
    -sr 16000
    ```
    Our experiments use the mTEDx_pt and g_neutral_speech_male datasets and automatically resample all audios to 16kHz. More detailed insctructions on how to download and process the datasets can be found in [TODO: add data processing scripts]

    The features will be extracted into a ``.lmdb`` file. Each entry in the file can be read similarly to a python dictionary where item corresponds to a type of feature. 

3. Set up the dataloader

    If you're implementing your own dataset, you will need to build a pytorch-style dataloader for it then add the dataset name in the training script.

    If you're using the already implemented datasets, simply use the dataset name in the training config file.

4. Config file
    Specify all desired training and model parameters in a ``.json`` file inside ``/config``

5. Run training script
    ```
    python -m src.train --config <config_path>
    ```

6. Follow experiment
    Training and evaluation metrics, as well as checkpoints, will be saved under ``/exp`` with the ``exp_name`` as defined in the used config file.