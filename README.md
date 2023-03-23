# NLP project about CANINE

[![Build & Test](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/main.yml/badge.svg)](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/main.yml)
[![Code quality](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/quality.yml/badge.svg)](https://github.com/gwatkinson/mva_snlp_canine/actions/workflows/quality.yml)
[![codecov](https://codecov.io/github/gwatkinson/mva_snlp_canine/branch/main/graph/badge.svg)](https://codecov.io/gh/gwatkinson/mva_snlp_canine)
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

2. Install the project
- With `poetry` ([installation](https://python-poetry.org/docs/#installation)):
```bash
poetry install
```
- With `pip` :
```bash
pip install -e .
```

3. Install pre-commit
```bash
pre-commit install
```

4. Use Pytorch with GPU support (optional). Use this if Pytorch doesn't see your GPU. This reinstalls Pytorch in the virtual environment, and needs to be rerun after each modification of the environment.
```bash
poe torch_cuda
```

## Reproduce the experiments

### Natural Language Inference (NLI)

In this section, we will describe how to reproduce the experiments for the NLI task.

#### Download and preprocess the dataset

We used the xnli dataset from the [HuggingFace datasets](https://huggingface.co/datasets/xnli) library.

But to keep it simple, we used only a sample of the total dataset, and a selection of languages. We also kept some low ressource languages in the test split, not seen in the training and validation splits.

To download and preprocess the dataset, run the following command :
```bash
nli_process_data EXPERIMENT_NAME
```

Usage:
```bash
Usage: nli_process_data [OPTIONS] EXPERIMENT_NAME

Options:
  -nt, --num_train_samples INTEGER    Number of training samples.  [default: 30000]
  -nv, --num_val_samples INTEGER      Number of validation samples.  [default: 1500]
  -ns, --num_test_samples INTEGER     Number of test samples.  [default: 2000]
  -tls, --train_language_subset LIST  Languages to keep in the train and validation set.
                                      [default: en, fr, ar, hi, el, ru, tr, zh]
  -tlp, --train_probs LIST            Probabilities of the languages to keep in the train and validation set.
                                      [default: 0.5, 0.3, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]
  -sls, --test_language_subset LIST   Languages to keep in the test set.
                                      [default: en, fr, es, bg, th, ur, sw]
  -slp, --test_probs LIST             Probabilities of the languages to keep in the test set.
                                      [default: 0.5, 0.3, 0.075, 0.05, 0.025, 0.025, 0.025]
  -o, --save_path TEXT                Path to save the dataset.  [default: data/nli/processed_dataset]
  -h, --hub_path TEXT                 Path to the dataset on the HuggingFace Hub. [default: Gwatk/xnli_subset]
  --seed INTEGER                      Seed for the random number generator. [default: 123]
  -j, --n_jobs INTEGER                Number of jobs to run in parallel. [default: 12]
  --no_pbar BOOLEAN                   Disable the progress bar.  [default: False]
  --token TEXT                        Token to login to the hub.  [default:None]
  --help                              Show this message and exit.
```


#### Tokenize the dataset using different models

We used the [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/python/latest/) library to tokenize the dataset, and create datasets containing the tokenized inputs for different models.


To tokenize the previously created dataset, run the following command :
```bash
nli_tokenize_data EXPERIMENT_NAME
```

Usage:
```bash
Usage: nli_tokenize_data [OPTIONS] EXPERIMENT_NAME

Options:
  -d, --dataset_name_or_path TEXT   Name or path of the dataset to tokenize.
                                    [default: Gwatk/{experiment_name}_xnli_subset]
  -m, --model_list LIST             List of models to use for tokenization.
                                    [default: bert-base-multilingual-cased, google/canine-s, google/canine-c]
  -p, --model_postfix LIST          List of prefixes to use for the tokenized datasets.
                                    [default: bert, canine_s, canine_c]
  -o, --save_path TEXT              Path to save the dataset.
                                    [default: nli_results/{experiment_name}/data/tokenized/{postfix}]
  -h, --hub_path TEXT               Path to the dataset on the HuggingFace Hub.
                                    [default: Gwatk/{experiment_name}_xnli_subset_tokenized_{postfix}]
  -j, --n_jobs INTEGER              Number of jobs to run in parallel. [default: 12]
  --no_pbar BOOLEAN                 Disable the progress bar.  [default: False]
  --token TEXT                      Token to login to the hub.  [default:None]
  --help                            Show this message and exit.
```
