"""Default values for the NLI dataset processing.

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
EXPERIMENT_NAME = "50k_eu_ru"
SEED = 123
N_JOBS = 12
NO_PBAR = False
SAVE_LOCAL = True
PUSH_TO_HUB = True
TOKEN = get_token(path=None)
DATASET_IS_TOKENISED = False

# Default paths
DIR_PATH_PREPROCESSED_DATASET = f"nli_results/{EXPERIMENT_NAME}/data/processed_dataset"
DIR_PATH_TOKENIZED_DATASET = (
    f"nli_results/{EXPERIMENT_NAME}/data/tokenized/" + "{postfix}"
)
DIR_TEMPLATE_TRAINING = f"nli_results/{EXPERIMENT_NAME}/models/" + "{postfix}"

# HuggingFace Hub paths
HUB_PATH_PREPROCESSED_DATASET = f"Gwatk/{EXPERIMENT_NAME}_xnli_subset"
HUB_PATH_TOKENIZER_DATASET = f"Gwatk/{EXPERIMENT_NAME}_tokenized" + "_{postfix}"
HUB_TEMPLATE_TRAINING = f"Gwatk/{EXPERIMENT_NAME}_nli_finetuned" + "_{postfix}"


# Default values for the NLI dataset processing
# Options: ["en", "ar", "fr", "es", "de", "el", "bg", "ru", "tr", "zh", "th", "vi", "hi", "ur", "sw"]
TRAIN_LANGUAGES_SUBSET = ["en", "fr", "de", "el", "ru"]
TRAIN_PROBS = [0.2, 0.2, 0.2, 0.2, 0.2]

TEST_LANGUAGES_SUBSET = ["en", "fr", "de", "ru", "el", "es", "bg"]
TEST_PROBS = [0.125, 0.125, 0.125, 0.125, 0.125, 0.1875, 0.1875]

NUM_TRAIN_SAMPLES = 50000
NUM_VAL_SAMPLES = 2000
NUM_TEST_SAMPLES = 5000


# Default values for the NLI models to use for tokenization and finetuning
MODEL_LIST = ["bert-base-multilingual-cased", "google/canine-s", "google/canine-c"]
MODEL_POSTFIX = ["bert", "canine_s", "canine_c"]


# Default values for the NLI finetuning
NUM_LABELS = 3

TRAINING_KWARGS = {
    # Training parameters
    "do_train": True,
    "do_eval": True,
    "fp16": True,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "torch_compile": False,
    "overwrite_output_dir": False,
    # Logging
    "logging_strategy": "steps",
    "logging_first_step": True,
    "logging_steps": 100,
    # Saving strategy
    "save_strategy": "epoch",
    # Evaluation strategy
    "evaluation_strategy": "epoch",
    "metric_for_best_model": "eval_accuracy",
    "greater_is_better": True,
}
