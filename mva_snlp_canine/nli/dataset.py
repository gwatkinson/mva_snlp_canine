import os

import click
import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def change_hypothesis_format(example, language_subset):
    print(example["hypothesis"])
    example["hypothesis_dict"] = {
        k: v
        for k, v in zip(
            example["hypothesis"]["language"], example["hypothesis"]["translation"]
        )
        if k in language_subset
    }
    return example


def choose_language(
    example,
    languages,
    probs,
):
    from numpy.random import choice

    lang = choice(languages, p=probs)
    example["language"] = lang
    example["choosen_premise"] = example["premise"][lang]
    example["choosen_hypothesis"] = example["hypothesis_dict"][lang]

    return example


def tokenize_example(example, tokenizer):
    return tokenizer(
        text=example["choosen_premise"],
        text_pair=example["choosen_hypothesis"],
        truncation=True,
    )


def process_dataset(
    num_train_samples: int,
    num_val_samples: int,
    num_test_samples: int,
    train_language_subset: list[str],
    train_probs: list[float],
    test_language_subset: list[str],
    test_probs: list[float],
    save_path: str,
    push_to_hub: bool,
    hub_path: str,
    seed: int,
    n_jobs: int,
):
    full_dataset = load_dataset("xnli", "all_languages")

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

    for phase in tqdm(["train", "validation", "test"]):
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

    dataset = dataset.select_columns(
        ["language", "choosen_premise", "choosen_hypothesis", "label"]
    )

    dataset.save_to_disk(save_path)

    if push_to_hub:
        login()
        dataset.push_to_hub(hub_path)  # hub_path = "Gwatk/xnli_subset"

    return dataset


def tokenize_dataset(
    dataset_name_or_path: str,
    model_name_or_path: str,
    save_path: str,
    push_to_hub: bool,
    hub_path: str,
    n_jobs: int,
):
    dataset = load_dataset(dataset_name_or_path)  # hub_path = "Gwatk/xnli_subset"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    dataset = dataset.select_columns(["choosen_premise", "choosen_hypothesis", "label"])

    for phase in tqdm(["train", "validation", "test"]):
        dataset[phase] = dataset[phase].map(
            tokenize_example, num_proc=n_jobs, fn_kwargs={"tokenizer": tokenizer}
        )

    dataset.save_to_disk(save_path)

    if push_to_hub:
        login()
        dataset.push_to_hub(
            hub_path
        )  # hub_path = "Gwatk/xnli_subset_canine-c_tokenized"
