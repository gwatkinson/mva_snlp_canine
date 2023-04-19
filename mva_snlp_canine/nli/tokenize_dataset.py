from pathlib import Path
from typing import Any

import click
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from mva_snlp_canine.nli.utils import load_config_nli, load_dataset_from_config


def tokenize_example(example: Any, tokenizer: Any, max_length: int):
    """Tokenize the example.

    Args:
        example (Any): The example to process from the dataset.
        tokenizer (Any): The tokenizer to use.

    Returns:
        example: The processed example.
    """
    tmp = tokenizer(
        text=example["choosen_premise"],
        text_pair=example["choosen_hypothesis"],
        truncation="only_first",
        max_length=max_length,
    )
    return tmp


def tokenize_dataset(
    dataset: Any,
    model_name_or_path: str,
    save_path: str or None,
    hub_path: str or None,
    n_jobs: int,
    no_pbar: bool,
    save_local: bool,
    push_to_hub: bool,
    token: str or None,
):
    """Tokenize the dataset and save it.

    Args:
        dataset (Any): Dataset to tokenize.
        model_name_or_path (str): Name or path of the model to use for the tokenization.
        save_path (str or None): Path to save the tokenized dataset. If None, the dataset is not saved.
        hub_path (str or None): Path to save the tokenized dataset on the hub. If None, the dataset is not pushed to the hub.
        n_jobs (int): Number of processes to use for the transformations when possible.
        no_pbar (bool): Whether to display the progress bars.
        save_local (bool): Whether to save the dataset locally.
        push_to_hub (bool): Whether to push the dataset to the hub.
        token (str or None): Token to login to the hub.
    """
    # Load the dataset and the tokenizer
    print("\n--- Loading the tokenizer...")
    # dataset = load_dataset(dataset_name_or_path)  # hub_path = "Gwatk/xnli_subset"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_length = tokenizer.model_max_length // 2

    dataset = dataset.select_columns(["choosen_premise", "choosen_hypothesis", "label"])

    # Tokenize the dataset
    print("\n--- Tokenizing the dataset...")
    for phase in tqdm(["train", "validation", "test"], disable=no_pbar):
        dataset[phase] = dataset[phase].map(
            tokenize_example,
            num_proc=n_jobs,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        )

    dataset = dataset.select_columns(
        ["input_ids", "attention_mask", "token_type_ids", "label"]
    )

    if save_local:
        print(f"\n--- Saving the dataset to disk to {save_path}...")
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(save_path)

    if push_to_hub:
        print(f"\n--- Pushing the dataset to the hub to {hub_path}...")
        login(token=token)
        dataset.push_to_hub(hub_path)

    return dataset


@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config_nli(config_file_path)

    dataset = load_dataset_from_config(cfg, "tokenized")

    model_list = cfg.MODEL_LIST
    model_postfix = cfg.MODEL_POSTFIX
    for model_name_or_path, postfix in zip(model_list, model_postfix):
        print(f"Tokenizing {model_name_or_path}...")

        model_token_save_path = cfg.DIR_PATH_TOKENIZED_DATASET.format(postfix=postfix)
        model_token_hub_save_path = cfg.HUB_PATH_TOKENIZER_DATASET.format(
            postfix=postfix
        )

        tokenize_dataset(
            dataset=dataset,
            model_name_or_path=model_name_or_path,
            save_path=model_token_save_path,
            hub_path=model_token_hub_save_path,
            n_jobs=cfg.N_JOBS,
            no_pbar=cfg.NO_PBAR,
            save_local=cfg.SAVE_LOCAL,
            push_to_hub=cfg.PUSH_TO_HUB,
            token=cfg.TOKEN,
        )


if __name__ == "__main__":
    main()
