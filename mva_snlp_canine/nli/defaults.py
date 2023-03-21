"""Default values for the NLI dataset processing."""

TRAIN_LANGUAGES_SUBSET = ["en", "fr", "ar", "hi", "el", "ru", "tr", "zh"]
TRAIN_PROBS = [0.5, 0.3, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]

TEST_LANGUAGES_SUBSET = ["en", "fr", "es", "bg", "th", "ur", "sw"]
TEST_PROBS = [0.5, 0.3, 0.075, 0.05, 0.025, 0.025, 0.025]

HUB_PATH_SUBSET = "Gwatk/xnli_subset"
HUB_PATH_TOKENIZED_CANINE_C = "Gwatk/xnli_subset_tokenized_canine_c"
HUB_PATH_TOKENIZED_CANINE_S = "Gwatk/xnli_subset_tokenized_canine_s"
HUB_PATH_TOKENIZED_BERT = "Gwatk/xnli_subset_tokenized_bert"