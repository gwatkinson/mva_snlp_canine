import json
from pathlib import Path

import click

from mva_snlp_canine.nli.defaults import (
    MODEL_LIST,
    MODEL_POSTFIX,
    NUM_LABELS,
    TOKEN,
    TOKENIZED_HUB_PATH,
    TRAINING_HUB_PATH,
    TRAINING_KWARGS_PATH,
    TRAINING_OUTPUT_DIR,
)
from mva_snlp_canine.nli.training import finetune_model


@click.command()
@click.argument("experiment_name", type=str)
@click.option(
    "--dataset_name_or_path",
    "-d",
    default=TOKENIZED_HUB_PATH,
    type=str,
    help="Name or path of the tokenized dataset. Used with format() to get the path to the datasets.",
    show_default=True,
)
@click.option(
    "--model_list",
    "-m",
    default=MODEL_LIST,
    type=list[str],
    help="List of models to use for trainig.",
    show_default=True,
)
@click.option(
    "--model_postfix",
    "-p",
    default=MODEL_POSTFIX,
    type=list[str],
    help="List of prefixes to use for the models.",
    show_default=True,
)
@click.option(
    "--num_labels",
    "-n",
    default=NUM_LABELS,
    type=int,
    help="Number of labels in the classification task.",
    show_default=True,
)
@click.option(
    "--output_dir",
    "-o",
    default=TRAINING_OUTPUT_DIR,
    type=str,
    help="Path to save the results of the training. Used with format.",
    show_default=True,
)
@click.option(
    "--training_kwargs_path",
    "-k",
    default=TRAINING_KWARGS_PATH,
    type=str,
    help="Path to the json file containing the training arguments.",
    show_default=True,
)
@click.option(
    "--hub_path",
    "-h",
    default=TRAINING_HUB_PATH,
    type=str or None,
    help="Path to the model on the HuggingFace Hub.",
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
    num_labels,
    output_dir,
    training_kwargs_path,
    hub_path,
    token,
):
    args_file = Path(training_kwargs_path)
    if args_file.exists():
        with args_file.open() as f:
            training_kwargs = json.load(f)
    else:
        raise ValueError(f"File {args_file} does not exist.")
    print(f"--- Loading training arguments from {training_kwargs_path}")
    print(f"--- Training arguments: {training_kwargs}")

    for model_name_or_path, postfix in zip(model_list, model_postfix):
        print(f"Training for {model_name_or_path}...")

        tokenized_dataset_path = dataset_name_or_path.format(
            experiment_name=experiment_name, postfix=postfix
        )
        experiment_output_dir = output_dir.format(
            experiment_name=experiment_name, postfix=postfix
        )
        if hub_path is not None:
            experiment_hub_path = hub_path.format(
                experiment_name=experiment_name, postfix=postfix
            )
        else:
            experiment_hub_path = None

        finetune_model(
            model_name_or_path=model_name_or_path,
            dataset_name_or_path=tokenized_dataset_path,
            num_labels=num_labels,
            output_dir=experiment_output_dir,
            training_kwargs=training_kwargs,
            hub_path=experiment_hub_path,
            token=token,
        )


if __name__ == "__main__":
    main()
