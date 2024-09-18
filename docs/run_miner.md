# Miner Guide

Fine-tune the creative model and submit it to the subnet for rewards.

## Install dependencies

First of all, you should install dependencies to run the following steps.


(Optional) It is recommended to create a new virtual environment to install dependencies.

```bash
python -m venv venv
source venv/bin/activate
```

Using this command to install all dependencies:

```bash
pip install -r requirements.txt && pip install .
```

## Prepare dataset

The dataset for finetuning are audio files. You should put all your training audio files in a single directory.
The supported file extensions can be WAV, mp3, ogg, flv or [anything else ffmpeg supports](https://ffmpeg.org//general.html#File-Formats).
But file extensions of your dataset should be all the same.

## Preprocess dataset

You should preprocess your dataset before finetuning.

Using the following command to preprocess your dataset:

```bash
python -m genki.model_trainer.preprocess --src <your-original-dataset-path> --dst <preporcessed-dataset-path>
```

Here are some options of this command:

* --src: Source dataset files directory
* --dst: Preprocessed dataset files directory
* --duration: Minimum audio length threshold in seconds. Audio files shorter than this threshold will be excluded. Default is 30
* --ext: Expected file extension of audio files. Default is wav

The preprocessed dataset files will be put in directory specified by option `dst`.
In the preprocessed dataset, you will audio files in wav format and json files with the same name of its corresponding audio file.
The json file contains some attributes of its corresponding audio. These attributes are generated automatically, and you can edit them maually. For example, you can edit the `description` attribute in json file to improve its quality.

## Finetuning

After preprocessing your dataset, you can use it for finetuning.

Using the following command to finetune your audio model:

```bash
python -m genki.model_trainer.train --dataset_path <preporcessed-dataset-path>
```

Here are some options of this command:

* --dataset_path: Training dataset path
* --checkpoint_path: Path to save the checkpoints. Default is checkpoints.
* --save_path: Path to save the finetuned model. Default is models
* --model_id: Base audiocraft model id to finetune. Default is facebook/musicgen-small
* --lr: Initial learning rate. Default is 1e-4.
* --epochs: Total training epochs. Default is 100.
* --use_wandb: Whether to use wandb to log. If this option is set, means use wandb.
* --save_epoch: Interval epochs to save checkpoints. Default is 10.
* --weight_decay: Weight decay. Default is 1e-4.
* --grad_acc: Gradiant accumulation steps. Default is 2.
* --warmup_steps: Gradient scheduler warmup steps. Default is 16.
* --batch_size: Batch size. Default is 4.
* --no_scaler: Whether to disable gradient scaler. If this option is set, means disable gradient scaler.

After running this command, you will see some logs printed in the terminal. The logs contains the current training epoch and the loss on the training dataset.

After training is completed, the finetuned model file will be saved in the directory specified by option `save_path`. You can use command to upload this model to a huggingface repo.

## Upload finetuned model

After finetuning, you need to upload your finetuned model to a huggingface repo to join a subnet competition.

Before uploading, you need to create a huggingface account and create a repo on it first. You can follow [this guide](https://huggingface.co/docs/hub/repositories-getting-started) to create a model type repo. You also need to generate a user access token to be able to upload local files to your remote repo. You can follow [this guide](https://huggingface.co/docs/hub/security-tokens) to generate a user access token.

Now you can use this command to upload your finetuned model to your huggingface repo:

```bash
HF_ACCESS_TOKEN=<your-user-access-token> python -m genki.model_trainer.upload --repo_id <your-huggingface-repo-id> --src <your-local-finetuned-model-path>
```

The option `repo_id` should be your huggingface repo id. The option `src` should be the local path of your finetuned model. It should be the same as the `save_path` option the finetuning command. You should set your huggingface user access token by environment variable `HF_ACCESS_TOKEN`.
