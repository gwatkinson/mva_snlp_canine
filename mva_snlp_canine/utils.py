from importlib import import_module
from pathlib import Path

from datasets import load_dataset
from huggingface_hub.constants import HF_TOKEN_PATH


def get_token(path: str = HF_TOKEN_PATH):
    """Get the token from the environment variable."""
    token_path = Path(path)
    if token_path.exists():
        print(f"---Found token at {path}.")
        return token_path.read_text()
    else:
        print(f"---No token found at {path}.")
        return None


def check_config_nli(config):
    """Check if the needed variables are defined in the config."""
    attribute_list = [
        "EXPERIMENT_NAME",
        "SEED",
        "N_JOBS",
        "NO_PBAR",
        "SAVE_LOCAL",
        "PUSH_TO_HUB",
        "TOKEN",
        "DATASET_IS_TOKENISED",
        "DIR_PATH_PREPROCESSED_DATASET",
        "DIR_PATH_TOKENIZED_DATASET",
        "DIR_TEMPLATE_TRAINING",
        "HUB_PATH_PREPROCESSED_DATASET",
        "HUB_PATH_TOKENIZER_DATASET",
        "HUB_TEMPLATE_TRAINING",
        "TRAIN_LANGUAGES_SUBSET",
        "TRAIN_PROBS",
        "TEST_LANGUAGES_SUBSET",
        "TEST_PROBS",
        "NUM_TRAIN_SAMPLES",
        "NUM_VAL_SAMPLES",
        "NUM_TEST_SAMPLES",
        "MODEL_LIST",
        "MODEL_POSTFIX",
        "NUM_LABELS",
        "TRAINING_KWARGS",
    ]
    missing_attributes = []
    for attribute in attribute_list:
        if not hasattr(config, attribute):
            print(f"{attribute} does not exist")
            missing_attributes.append(attribute)
    return missing_attributes


def load_config_nli(config_file_path):
    """Load the config file."""
    module = import_module(config_file_path)
    missing_attributes = check_config_nli(module)
    assert len(missing_attributes) == 0, f"Missing attributes: {missing_attributes}"
    return module


def load_dataset_from_config(cfg, kind, postfix=None):
    if kind == "preprocessed":
        dataset_path = cfg.DIR_PATH_PREPROCESSED_DATASET
    elif kind == "tokenized":
        dataset_path = cfg.DIR_PATH_TOKENIZED_DATASET.format(postfix=postfix)
    try:
        print(f"--- Loading dataset: {dataset_path}...")
        dataset = load_dataset(dataset_path)
    except FileNotFoundError:
        print(f"--- Dataset not found: {dataset_path}...")
        try:
            if kind == "preprocessed":
                dataset_path = cfg.HUB_PATH_PREPROCESSED_DATASET
            elif kind == "tokenized":
                dataset_path = cfg.HUB_PATH_TOKENIZER_DATASET.format(postfix=postfix)
            print(f"--- Loading dataset: {dataset_path}...")
            dataset = load_dataset(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"--- Dataset not found: {dataset_path}...")

    return dataset
