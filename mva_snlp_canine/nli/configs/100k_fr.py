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

from mva_snlp_canine.nli.utils import get_token

# General parameters
EXPERIMENT_NAME = "100k_fr"
RESULTS_FOLDER = "nli_results"
HUGGINGFACE_USERNAME = "Gwatk"
SEED = 123
N_JOBS = 12
NO_PBAR = False
SAVE_LOCAL = True
PUSH_TO_HUB = False
TOKEN = get_token()
DATASET_IS_TOKENISED = False

# Default paths
DIR_PATH_PREPROCESSED_DATASET = (
    f"{RESULTS_FOLDER}/{EXPERIMENT_NAME}/data/processed_dataset"
)
DIR_PATH_TOKENIZED_DATASET = (
    f"{RESULTS_FOLDER}/{EXPERIMENT_NAME}/data/tokenized/" + "{postfix}"
)
DIR_TEMPLATE_TRAINING = f"{RESULTS_FOLDER}/{EXPERIMENT_NAME}/models/" + "{postfix}"

# HuggingFace Hub paths
HUB_PATH_PREPROCESSED_DATASET = f"{HUGGINGFACE_USERNAME}/{EXPERIMENT_NAME}_xnli_subset"
HUB_PATH_TOKENIZER_DATASET = (
    f"{HUGGINGFACE_USERNAME}/{EXPERIMENT_NAME}_tokenized" + "{postfix}"
)
HUB_TEMPLATE_TRAINING = (
    f"{HUGGINGFACE_USERNAME}/{EXPERIMENT_NAME}_nli_finetuned" + "{postfix}"
)

# language_to_abbr = {
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
# }
# Default values for the NLI dataset processing
# Options: ["en", "ar", "fr", "es", "de", "el", "bg", "ru", "tr", "zh", "th", "vi", "hi", "ur", "sw"]
TRAIN_LANGUAGES_SUBSET = ["fr"]
TRAIN_PROBS = [1.0]

TEST_LANGUAGES_SUBSET = ["fr"]
TEST_PROBS = [1.0]

NUM_TRAIN_SAMPLES = 100_000
NUM_VAL_SAMPLES = 2490
NUM_TEST_SAMPLES = 100


#  Default values for the NLI models to use for tokenization and finetuning
# MODEL_LIST = ["google/canine-s", "google/canine-c", "bert-base-multilingual-cased", "camembert-base"]
# MODEL_POSTFIX = ["canine_s", "canine_c", "bert", "camembert"]
MODEL_LIST = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
MODEL_POSTFIX = ["MiniLM"]


# Default values for the NLI finetuning
NUM_LABELS = 3

TRAINING_KWARGS = {
    # Optimizer parameters
    "do_train": True,
    "do_eval": True,
    "fp16": True,
    "auto_find_batch_size": False,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    # Training parameters
    "num_train_epochs": 5,
    "per_device_train_batch_size": 6,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
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
}
