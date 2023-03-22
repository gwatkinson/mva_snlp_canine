"""Default values for the NLI dataset processing."""

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

PREPROCESSING_PATH = "data/nli/processed_dataset"
PREPROCESSING_HUB_PATH = "Gwatk/xnli_subset"

TOKENIZED_HUB_PATH = "Gwatk/xnli_subset_tokenized_{postfix}"
TOKENIZED_PATH = "data/nli/tokenized/{postfix}"

MODEL_LIST = ["bert-base-multilingual-cased", "google/canine-s", "google/canine-c"]
MODEL_POSTFIX = ["bert", "canine_s", "canine_c"]


N_EPOCHS = 3
NUM_LABELS = 3

MODEL_OUTPUT_PATH = "models/nli/{postfix}"
