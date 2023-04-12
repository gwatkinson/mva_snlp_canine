from pathlib import Path
from typing import Any

import click
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from tqdm.auto import tqdm

from mva_snlp_canine.utils import load_config_nli


# Utility functions
def change_hypothesis_format(example: Any, language_subset: list[str]):
    """Change the hypothesis format from a dictionaries of lists to a dictionary.

    Args:
        example (Any): The example to process from the dataset.
        language_subset (list[str]): The languages to keep.

    Returns:
        example: The processed example.
    """
    tmp = {
        "hypothesis_dict": {
            k: v
            for k, v in zip(
                example["hypothesis"]["language"], example["hypothesis"]["translation"]
            )
            if k in language_subset
        }
    }
    return tmp


def choose_language(
    example: Any,
    languages: list[str],
    probs: list[float],
):
    """Choose a language for the example.

    Args:
        example (Any): The example to process from the dataset.
        languages (list[str]): The languages to choose from.
        probs (list[float]): The probabilities of the languages to choose from. Must sum to 1.

    Returns:
        example: The processed example.
    """
    from numpy.random import choice

    lang = choice(languages, p=probs)
    tmp = {
        "language": lang,
        "choosen_premise": example["premise"][lang],
        "choosen_hypothesis": example["hypothesis_dict"][lang],
    }

    return tmp


def process_dataset(
    num_train_samples: int,
    num_val_samples: int,
    num_test_samples: int,
    train_language_subset: list[str],
    train_probs: list[float],
    test_language_subset: list[str],
    test_probs: list[float],
    save_path: str or None,
    hub_path: str or None,
    seed: int,
    n_jobs: int,
    no_pbar: bool,
    save_local: bool,
    push_to_hub: bool,
    token: str or None,
):
    """Apply the preprocessing transformations (sampling and language selection) to the dataset and save it.

    Args:
        num_train_samples (int): Number of samples to keep in the train set.
        num_val_samples (int): Number of samples to keep in the validation set.
        num_test_samples (int): Number of samples to keep in the test set.
        train_language_subset (list[str]): Languages to keep in the train and validation set.
        train_probs (list[float]): Probabilities of the languages to keep in the train and validation set.
        test_language_subset (list[str]): Languages to keep in the test set.
        test_probs (list[float]): Probabilities of the languages to keep in the test set.
        save_path (str or None): Save path of the dataset. If None, the dataset is not saved.
        hub_path (str or None): Save path of the dataset on the hub. If None, the dataset is not pushed to the hub.
        seed (int): Seed for the random sampling.
        n_jobs (int): Number of processes to use for the transformations when possible.
        no_pbar (bool): Whether to display the progress bars.
        save_local (bool): Whether to save the dataset locally.
        push_to_hub (bool): Whether to push the dataset to the hub.
        token (str or None): Token to login to the hub.

    Returns:
        dataset: The processed dataset.
    """
    # Load the dataset
    print("\n--- Loading the dataset...")
    full_dataset = load_dataset("xnli", "all_languages")

    # Shuffle and select a subset of the dataset
    print("\n--- Shuffling and selecting a subset of the dataset...")
    dataset = DatasetDict(
        {
            "train": full_dataset["train"]
            .shuffle(seed)
            .select(range(num_train_samples)),
            "validation": full_dataset["validation"]
            .shuffle(seed)
            .select(range(num_val_samples)),
            "test": full_dataset["test"].shuffle(seed).select(range(num_test_samples)),
        }
    )

    description_postfix = "This dataset is a subset of the XNLI dataset. It contains {num_samples} samples and only the following languages: {language_subset}, with the following probabilities: {probs}."

    # Apply the transformations and add the description
    print("\n--- Applying the transformations...")
    for phase in tqdm(["train", "validation", "test"], disable=no_pbar):
        num_samples = (
            num_train_samples
            if phase == "train"
            else num_val_samples
            if phase == "validation"
            else num_test_samples
        )
        language_subset = (
            test_language_subset if phase == "test" else train_language_subset
        )
        probs = test_probs if phase == "test" else train_probs

        dataset[phase] = dataset[phase].map(
            change_hypothesis_format,
            num_proc=n_jobs,
            fn_kwargs={"language_subset": language_subset},
        )
        dataset[phase] = dataset[phase].map(
            choose_language,
            num_proc=n_jobs,
            fn_kwargs={"languages": language_subset, "probs": probs},
        )
        dataset[phase].info.description += description_postfix.format(
            num_samples=num_samples, language_subset=language_subset, probs=probs
        )

    # Select the columns we want to keep
    dataset = dataset.select_columns(
        ["language", "choosen_premise", "choosen_hypothesis", "label"]
    )

    # Save the dataset
    if save_local:
        print(f"\n--- Saving the dataset to disk to {save_path}...")
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(save_path)

    if push_to_hub:
        print(f"\n--- Pushing the dataset to the hub to {hub_path}...")
        login(token=token)
        dataset.push_to_hub(hub_path)  # hub_path = "Gwatk/xnli_subset"

    return dataset


# CLI function
@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config_nli(config_file_path)

    process_dataset(
        num_train_samples=cfg.NUM_TRAIN_SAMPLES,
        num_val_samples=cfg.NUM_VAL_SAMPLES,
        num_test_samples=cfg.NUM_TEST_SAMPLES,
        train_language_subset=cfg.TRAIN_LANGUAGES_SUBSET,
        train_probs=cfg.TRAIN_PROBS,
        test_language_subset=cfg.TEST_LANGUAGES_SUBSET,
        test_probs=cfg.TEST_PROBS,
        save_path=cfg.DIR_PATH_PREPROCESSED_DATASET,
        hub_path=cfg.HUB_PATH_PREPROCESSED_DATASET,
        seed=cfg.SEED,
        n_jobs=cfg.N_JOBS,
        no_pbar=cfg.NO_PBAR,
        save_local=cfg.SAVE_LOCAL,
        push_to_hub=cfg.PUSH_TO_HUB,
        token=cfg.TOKEN,
    )


if __name__ == "__main__":
    main()
