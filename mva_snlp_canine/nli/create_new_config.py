from pathlib import Path

import click

CONFIG_TEMPLATE = '''"""Default values for the NLI dataset processing.

Variables:
    EXPERIMENT_NAME: Name of the experiment.
    SEED: Seed for the random number generator.
    N_JOBS: Number of jobs to use for multiprocessing.
    NO_PBAR: If True, disable the progress bar.
    SAVE_LOCAL: If True, save the processed dataset locally.
    PUSH_TO_HUB: If True, push the processed dataset to the HuggingFace Hub.
    TOKEN: Token to use to push the processed dataset to the HuggingFace Hub.
    DATASET_IS_TOKENISED: If True, add the tokenization step just before training.

    DIR_PATH_PREPROCESSED_DATASET: Local path to the preprocessed dataset.
    DIR_PATH_TOKENIZED_DATASET: Local path to the tokenized dataset.
    DIR_TEMPLATE_TRAINING: Template to use for the local path to the training results.

    HUB_PATH_PREPROCESSED_DATASET: HuggingFace Hub path to the preprocessed dataset.
    HUB_PATH_TOKENIZER_DATASET: HuggingFace Hub path to the tokenized dataset.
    HUB_TEMPLATE_TRAINING: Template to use for the HuggingFace Hub path to the training results.

    TRAIN_LANGUAGES_SUBSET: Subset of languages to use for training.
    TRAIN_PROBS: Proportions of each language to use for training.
    TEST_LANGUAGES_SUBSET: Subset of languages to use for testing.
    TEST_PROBS: Proportions of each language to use for testing.

    NUM_TRAIN_SAMPLES: Number of samples to use for training.
    NUM_VAL_SAMPLES: Number of samples to use for validation.
    NUM_TEST_SAMPLES: Number of samples to use for testing.

    MODEL_LIST: List of models to use for tokenization and finetuning.
    MODEL_POSTFIX: List of postfixes to use for the local paths and HuggingFace Hub paths.

    NUM_LABELS: Number of labels in the NLI dataset.

    TRAINING_KWARGS: Default values for the training parameters.
"""

from mva_snlp_canine.utils import get_token

# General parameters
EXPERIMENT_NAME = '{experiment_name}'
RESULTS_FOLDER = 'nli_results'
HUGGINGFACE_USERNAME = '{huggingface_username}'
SEED = 123
N_JOBS = 12
NO_PBAR = False
SAVE_LOCAL = {save_local}
PUSH_TO_HUB = {push_to_hub}
TOKEN = get_token()
DATASET_IS_TOKENISED = False

# Default paths
DIR_PATH_PREPROCESSED_DATASET = f"{{RESULTS_FOLDER}}/{{EXPERIMENT_NAME}}/data/processed_dataset"
DIR_PATH_TOKENIZED_DATASET = (
    f"{{RESULTS_FOLDER}}/{{EXPERIMENT_NAME}}/data/tokenized/" + "{{postfix}}"
)
DIR_TEMPLATE_TRAINING = f"{{RESULTS_FOLDER}}/{{EXPERIMENT_NAME}}/models/" + "{{postfix}}"

# HuggingFace Hub paths
HUB_PATH_PREPROCESSED_DATASET = f"{{HUGGINGFACE_USERNAME}}/{{EXPERIMENT_NAME}}_xnli_subset"
HUB_PATH_TOKENIZER_DATASET = f"{{HUGGINGFACE_USERNAME}}/{{EXPERIMENT_NAME}}_tokenized" + "{{postfix}}"
HUB_TEMPLATE_TRAINING = f"{{HUGGINGFACE_USERNAME}}/{{EXPERIMENT_NAME}}_nli_finetuned" + "{{postfix}}"

# language_to_abbr = {{
#     "english": "en",
#     "arabic": "ar",
#     "french": "fr",
#     "spanish": "es",
#     "german": "de",
#     "greek": "el",
#     "bulgarian": "bg",
#     "russian": "ru",
#     "turkish": "tr",
#     "chinese": "zh",
#     "thai": "th",
#     "vietnamese": "vi",
#     "hindi": "hi",
#     "urdu": "ur",
#     "swahili": "sw",
# }}
# Default values for the NLI dataset processing
# Options: ["en", "ar", "fr", "es", "de", "el", "bg", "ru", "tr", "zh", "th", "vi", "hi", "ur", "sw"]
TRAIN_LANGUAGES_SUBSET = ["en"]
TRAIN_PROBS = [1.0]

TEST_LANGUAGES_SUBSET = ["en"]
TEST_PROBS = [1.0]

NUM_TRAIN_SAMPLES = {num_train_samples}
NUM_VAL_SAMPLES = {num_val_samples}
NUM_TEST_SAMPLES = {num_test_samples}


#  Default values for the NLI models to use for tokenization and finetuning
MODEL_LIST = ["google/canine-s", "google/canine-c", "bert-base-multilingual-cased"]
MODEL_POSTFIX = ["canine_s", "canine_c", "bert"]


# Default values for the NLI finetuning
NUM_LABELS = 3

TRAINING_KWARGS = {{
    # Optimizer parameters
    "do_train": True,
    "do_eval": True,
    "fp16": {fp16},
    "auto_find_batch_size": False,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    # Training parameters
    "num_train_epochs": {num_train_epochs},
    "per_device_train_batch_size": {batch_size},
    "per_device_eval_batch_size": {batch_size},
    "gradient_accumulation_steps": {gradient_accumulation_steps},
    "eval_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "torch_compile": False,
    "overwrite_output_dir": True,
    # Logging
    "run_name": "first_run",
    "logging_strategy": "steps",
    "logging_first_step": True,
    "logging_steps": 100,
    # Saving strategy
    "save_strategy": "epoch",
    # Evaluation strategy
    "evaluation_strategy": "epoch",
    "metric_for_best_model": "eval_accuracy",
    "greater_is_better": True,
}}'''


# click command that fills in the template with cli values with sensible defaults. Everything is optional and should be interactive in the cli
# add prompts to the cli to make it interactive
@click.command(
    help="Command that creates a config file to run an experiment.",
    no_args_is_help=True,
)
@click.argument("experiment_name")
@click.option(
    "--save_local",
    default=True,
    help="Save the processed dataset locally",
    show_default=True,
    prompt=True,
)
@click.option(
    "--push_to_hub",
    default=False,
    help="Push the processed dataset to the HuggingFace Hub",
    show_default=True,
    prompt=True,
)
@click.option(
    "--huggingface_username",
    default="Gwatk",
    help="HuggingFace username",
    show_default=True,
    prompt=True,
)
@click.option(
    "--num_train_samples",
    default=300_000,
    help="Number of samples to use for training",
    show_default=True,
    prompt=True,
)
@click.option(
    "--num_val_samples",
    default=3000,
    help="Number of samples to use for validation",
    show_default=True,
    prompt=True,
)
@click.option(
    "--num_test_samples",
    default=5000,
    help="Number of samples to use for testing",
    show_default=True,
    prompt=True,
)
@click.option(
    "--num_train_epochs",
    default=5,
    help="Number of training epochs",
    show_default=True,
    prompt=True,
)
@click.option(
    "--batch_size", default=8, help="Batch size", show_default=True, prompt=True
)
@click.option(
    "--gradient_accumulation_steps",
    default=4,
    help="Number of gradient accumulation steps",
    show_default=True,
    prompt=True,
)
@click.option(
    "--fp16",
    default=False,
    help="Whether to use mixed-precision training",
    show_default=True,
    prompt=True,
)
def main(
    experiment_name,
    save_local,
    push_to_hub,
    huggingface_username,
    num_train_samples,
    num_val_samples,
    num_test_samples,
    num_train_epochs,
    batch_size,
    gradient_accumulation_steps,
    fp16,
):
    # fill in the template with the cli values
    config = CONFIG_TEMPLATE.format(
        experiment_name=experiment_name,
        save_local=save_local,
        push_to_hub=push_to_hub,
        huggingface_username=huggingface_username,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        num_test_samples=num_test_samples,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
    )
    output_path = Path("mva_snlp_canine/nli/configs")
    output_path.mkdir(parents=True, exist_ok=True)
    # write the config to a file
    with open(output_path / (experiment_name + ".py"), "w") as f:
        f.write(config)


if __name__ == "__main__":
    main()
