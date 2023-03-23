"""Default values for the NLI dataset processing."""

from pathlib import Path

from huggingface_hub.constants import HF_TOKEN_PATH

TRAIN_LANGUAGES_SUBSET = ["en", "fr", "ar", "hi", "el", "ru", "tr", "zh"]
TRAIN_PROBS = [0.5, 0.3, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]

TEST_LANGUAGES_SUBSET = ["en", "fr", "es", "bg", "th", "ur", "sw"]
TEST_PROBS = [0.5, 0.3, 0.075, 0.05, 0.025, 0.025, 0.025]

NUM_TRAIN_SAMPLES = 30000
NUM_VAL_SAMPLES = 1500
NUM_TEST_SAMPLES = 2000

SEED = 123
N_JOBS = 12
NO_PBAR = False

PREPROCESSING_PATH = "nli_results/{experiment_name}/data/processed_dataset"
PREPROCESSING_HUB_PATH = "Gwatk/{experiment_name}_xnli_subset"

TOKENIZED_HUB_PATH = "Gwatk/{experiment_name}_xnli_subset_tokenized_{postfix}"
TOKENIZED_PATH = "nli_results/{experiment_name}/data/tokenized/{postfix}"

MODEL_LIST = ["bert-base-multilingual-cased", "google/canine-s", "google/canine-c"]
MODEL_POSTFIX = ["bert", "canine_s", "canine_c"]


N_EPOCHS = 3
NUM_LABELS = 3

MODEL_OUTPUT_PATH = "nli_results/{experiment_name}/models/{postfix}"


token_path = Path(HF_TOKEN_PATH)
if token_path.exists():
    TOKEN = token_path.read_text()
else:
    TOKEN = None
