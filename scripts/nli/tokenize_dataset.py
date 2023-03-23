import click
from datasets import load_dataset

from mva_snlp_canine.nli.dataset import tokenize_dataset
from mva_snlp_canine.nli.defaults import (
    MODEL_LIST,
    MODEL_POSTFIX,
    N_JOBS,
    NO_PBAR,
    PREPROCESSING_HUB_PATH,
    TOKEN,
    TOKENIZED_HUB_PATH,
    TOKENIZED_PATH,
)


@click.command()
@click.argument("experiment_name", type=str)
@click.option(
    "--dataset_name_or_path",
    "-d",
    default=PREPROCESSING_HUB_PATH,
    type=str,
    help="Name or path of the dataset to tokenize.",
    show_default=True,
)
@click.option(
    "--model_list",
    "-m",
    default=MODEL_LIST,
    type=list[str],
    help="List of models to use for tokenization.",
    show_default=True,
)
@click.option(
    "--model_postfix",
    "-p",
    default=MODEL_POSTFIX,
    type=list[str],
    help="List of prefixes to use for the tokenized datasets.",
    show_default=True,
)
@click.option(
    "--save_path",
    "-o",
    default=TOKENIZED_PATH,
    type=str,
    help="Path to save the dataset.",
    show_default=True,
)
@click.option(
    "--hub_path",
    "-h",
    default=TOKENIZED_HUB_PATH,
    type=str,
    help="Path to the dataset on the HuggingFace Hub.",
    show_default=True,
)
@click.option(
    "--n_jobs",
    "-j",
    default=N_JOBS,
    type=int,
    help="Number of jobs to run in parallel.",
    show_default=True,
)
@click.option(
    "--no_pbar",
    default=NO_PBAR,
    type=bool,
    help="Disable the progress bar.",
    show_default=True,
)
@click.option(
    "--token",
    default=TOKEN,
    type=str,
    help="Token to login to the hub.",
    show_default=True,
)
def main(
    experiment_name,
    dataset_name_or_path,
    model_list,
    model_postfix,
    save_path,
    hub_path,
    n_jobs,
    no_pbar,
    token,
):
    dataset_path = dataset_name_or_path.format(experiment_name=experiment_name)
    print(f"--- Loading dataset: {dataset_path}...")
    dataset = load_dataset(dataset_path)

    for model_name_or_path, postfix in zip(model_list, model_postfix):
        print(f"Tokenizing {model_name_or_path}...")

        token_save_path = save_path.format(
            experiment_name=experiment_name, postfix=postfix
        )
        token_hub_path = hub_path.format(
            experiment_name=experiment_name, postfix=postfix
        )

        tokenize_dataset(
            dataset=dataset,
            model_name_or_path=model_name_or_path,
            save_path=token_save_path,
            hub_path=token_hub_path,
            n_jobs=n_jobs,
            no_pbar=no_pbar,
            token=token,
        )


if __name__ == "__main__":
    main()
