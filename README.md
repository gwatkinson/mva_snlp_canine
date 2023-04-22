# NLP project about CANINE

[![Build](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/main.yml/badge.svg)](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/main.yml)
[![Code quality](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/quality.yml/badge.svg)](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the repository associated with the NLP project for the MVA course __Algorithms for Speech and NLP__.

This is a group project realised by :

* Gabriel Watkinson
* Josselin Dubois
* Marine Astruc
* Javier Ramos Gutiérrez

## Installation

1. Clone the repository
```bash
git clone https://github.com/gwatkinson/mva_snlp_canine
```

2. Install the project and dependencies, creating a virtual environment with `poetry` (you need to [install poetry](https://python-poetry.org/docs/#installation) it beforehand):
```bash
poetry install
```

3. Activate the environment
```bash
source $(poetry env info --path)/bin/activate  # for linux
# & ((poetry env info --path) + "\Scripts\activate.ps1")  # for windows powershell
# poetry shell  # or this spawns a new shell
```

4. Install pre-commit
```bash
pre-commit install
```

5. Use Pytorch with GPU support (optional). Use this if Pytorch doesn't see your GPU. This reinstalls Pytorch in the virtual environment, and needs to be rerun after each modification of the environment.
```bash
poe torch_cuda
```

## Reproduce the experiments

### Natural Language Inference (NLI)

In this section, we will describe how to reproduce the experiments for the NLI task.

All the functions and configs used for those experiments are in the `mva_snlp_canine/nli` folder.

#### Creating a config file

The experiments can be configured from config files.

To generate a generic one with some prompts, you can run:

```bash
nli_create_config [OPTIONS] EXPERIMENT_NAME
```

Usage :

```bash
nli_create_config [OPTIONS] EXPERIMENT_NAME

  Command that creates a config file to run an experiment.

Options:
  --train_languages_subset TEXT   Languages to use for training  [default: en]
  --save_local BOOLEAN            Save the processed dataset locally
                                  [default: True]
  --push_to_hub BOOLEAN           Push the processed dataset to the
                                  HuggingFace Hub  [default: False]
  --huggingface_username TEXT     HuggingFace username  [default: Gwatk]
  --num_train_samples INTEGER     Number of samples to use for training
                                  [default: 300000]
  --num_val_samples INTEGER       Number of samples to use for validation
                                  [default: 2490]
  --num_test_samples INTEGER      Number of samples to use for testing
                                  [default: 5000]
  --num_train_epochs INTEGER      Number of training epochs  [default: 5]
  --learning_rate FLOAT           Learning rate  [default: 0.0001]
  --batch_size INTEGER            Batch size  [default: 8]
  --gradient_accumulation_steps INTEGER
                                  Number of gradient accumulation steps
                                  [default: 4]
  --fp16 BOOLEAN                  Whether to use mixed-precision training
                                  [default: True]
  --help                          Show this message and exit.
```

Then, you should look into the newly created file in the `mva_snlp_canine/nli/configs` folder and change additional options if needed (especially the training arguments).

#### Running an experiment

From a config file, you just need to run:

```bash
nli_run_experiment EXPERIMENT_NAME
```

This will train the models, and can be quite long.

A bash file in the `scripts` folder also exists, that reproduces all the experiments mentionned in our report:

```bash
source scripts/run_nli_exps.sh
```

#### Evaluating the experiments

Once the model is trained, to evaluate it on the test set and on all languages, run:

```bash
nli_evaluate_experiment EXPERIMENT_NAME
```

The associated script is:

```bash
source scripts/evaluate_nli_exps.sh
```

#### Generating some graphs from the metrics dataframe

The evaluation step returns a dataframe, to visualize the results, run:

```bash
nli_visualise_results [OPTIONS] EXP_NAME

Options:
  --num TEXT        Number of samples used in the training set, optional.
  --languages TEXT  Languages used in the training set, optional.
  --attacked        Whether to visualise attacked metrics, default fault.
  --help            Show this message and exit.
```

The associated script is:

```bash
source scripts/visualise_nli_results.sh
```

#### Evaluate the models on perturbed datasets

Lastly, we aslo used [`nlpaug`](https://github.com/makcedward/nlpaug) to generate some perturbed inputs and then look at how the model reacts.

To generate these datasets and evaluate, run:

```bash
nli_augmented_dataset [OPTIONS] EXP_NAME

  Evaluate the experiment in the given directory.

Options:
  --language_subset TEXT  The languages to evaluate the model on.  Options are
                          ["ar", "bg", "de", "el", "en", "es", "fr", "hi",
                          "ru", "sw", "th", "tr", "ur", "vi", "zh"]  [default:
                          en,fr]
  --help                  Show this message and exit.
```

Then you can use the previous command to generate plots, using the `--attacked` flag.

The associated script is:

```bash
source scripts/evaluate_nli_attacks.sh
```

#### Running all the experiments

Finally the following script, runs all the previous script in the right order:

```bash
source scripts/nli_run_all.sh
```
