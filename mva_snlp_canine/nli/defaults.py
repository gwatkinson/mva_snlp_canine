"""Default values for the NLI dataset processing."""

from pathlib import Path

from huggingface_hub.constants import HF_TOKEN_PATH

# Get the token from the environment variable
token_path = Path(HF_TOKEN_PATH)
if token_path.exists():
    TOKEN = token_path.read_text()
else:
    TOKEN = None


# General parameters
SEED = 123
N_JOBS = 12
NO_PBAR = False


# Default values for the NLI dataset processing
TRAIN_LANGUAGES_SUBSET = ["en", "fr", "ar", "hi", "el", "ru", "tr", "zh"]
TRAIN_PROBS = [0.5, 0.3, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]

TEST_LANGUAGES_SUBSET = ["en", "fr", "es", "bg", "th", "ur", "sw"]
TEST_PROBS = [0.5, 0.3, 0.075, 0.05, 0.025, 0.025, 0.025]

NUM_TRAIN_SAMPLES = 100000
NUM_VAL_SAMPLES = 3000
NUM_TEST_SAMPLES = 5000

PREPROCESSING_PATH = "nli_results/{experiment_name}/data/processed_dataset"
PREPROCESSING_HUB_PATH = "Gwatk/{experiment_name}_xnli_subset"


# Default values for the NLI models to use for tokenization and finetuning
MODEL_LIST = ["bert-base-multilingual-cased", "google/canine-s", "google/canine-c"]
MODEL_POSTFIX = ["bert", "canine_s", "canine_c"]
# MODEL_LIST = ["google/canine-s"]
# MODEL_POSTFIX = ["canine_s"]


# Default values for the NLI dataset tokenization
TOKENIZED_HUB_PATH = "Gwatk/{experiment_name}_xnli_subset_tokenized_{postfix}"
TOKENIZED_PATH = "nli_results/{experiment_name}/data/tokenized/{postfix}"


# Default values for the NLI finetuning
TRAINING_OUTPUT_DIR = "nli_results/{experiment_name}/models/{postfix}"
TRAINING_HUB_PATH = "Gwatk/{experiment_name}_xnli_subset_finetuned_{postfix}"
# TRAINING_HUB_PATH = None

NUM_LABELS = 3
TRAINING_KWARGS_PATH = "mva_snlp_canine/nli/default_training_args.json"

TRAINING_KWARGS = {
    "evaluation_strategy": "epoch",
    "overwrite_output_dir": False,
    "gradient_checkpointing": True,
    "fp16": False,
    "do_train": True,
    "do_eval": True,
    "learning_rate": 5e-4,
    "gradient_accumulation_steps": 8,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 3,
}
