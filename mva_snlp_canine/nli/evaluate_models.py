"""Module that evaluates the model on all languages contained on the test dataset, regardless of the language used to train the model."""

from typing import Any

# import evaluate
import torch

# from datasets import load_dataset
from evaluate.visualization import radar_plot
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer

from mva_snlp_canine.nli.tokenize_dataset import tokenize_dataset
from mva_snlp_canine.nli.train_models import compute_metrics

language_subset = [
    "en",
    "ar",
    "fr",
    "es",
    "de",
    "el",
    "bg",
    "ru",
    "tr",
    "zh",
    "th",
    "vi",
    "hi",
    "ur",
    "sw",
]


def evaluate_model(model_path: str, dataset: Any, dataset_is_tokenized: bool):
    if not dataset_is_tokenized:
        print("--- Tokenizing the dataset...")
        dataset = tokenize_dataset(
            dataset=dataset,
            model_name_or_path=model_path,
            save_path=None,
            hub_path=None,
            n_jobs=12,
            no_pbar=False,
            save_local=False,
            push_to_hub=False,
            token=None,
        )

    print(f"\n--- Loading the model {model_path}...")
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    print("\n--- Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=4)

    print(f"\n--- Cuda is available: {torch.cuda.is_available()}")

    # clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n--- Evaluating the model...")
    predictions, label_ids, metrics = trainer.predict(dataset)
    trainer.log_metrics("test", metrics)

    data = [
        {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
        {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
        {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
        {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6},
    ]
    model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
    plot = radar_plot(data=data, model_names=model_names)
    plot.show()


# for language in language_subset:
#     dataset = load_dataset("xnli", language, split="test")
#     print(f"--- Language {language}: {len(dataset)}")

#     dataset = dataset.rename_columns({
#         "premise": "choosen_premise",
#         "hypothesis": "choosen_hypothesis",
#         "label": "label",
#     })
