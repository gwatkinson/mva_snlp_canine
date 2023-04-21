"""Module that evaluates the model on all languages contained on the test dataset, regardless of the language used to train the model."""

import glob
import json
from pathlib import Path

import click
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from mva_snlp_canine.nli.tokenize_dataset import tokenize_example
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


def evaluate_model_on_language(model_path: str, language: str):
    """Evaluate the model on the test dataset of the given language."""
    model_name = model_path.split("/")[-2].strip("_")
    print(f"=== Language : {language} | Model : {model_name}")
    dataset = load_dataset("xnli", language, split="test")

    dataset = dataset.rename_columns(
        {
            "premise": "choosen_premise",
            "hypothesis": "choosen_hypothesis",
            "label": "label",
        }
    )

    print(f"--- Loading the model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    print("--- Tokenizing the dataset...")
    dataset = dataset.map(
        tokenize_example,
        num_proc=12,
        fn_kwargs={"tokenizer": tokenizer, "max_length": None},
    )

    dataset = dataset.select_columns(
        ["input_ids", "attention_mask", "token_type_ids", "label"]
    )

    print(f"--- Cuda is available: {torch.cuda.is_available()}")

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("--- Evaluating the model...")
    predictions, label_ids, metrics = trainer.predict(dataset)
    metrics["language"] = language
    metrics["model_path"] = model_path
    metrics["model"] = model_name

    return predictions, label_ids, metrics


def evaluate_experiment(exp_dir: str):
    print(f"=== Evaluating the experiment {exp_dir}...")

    trainer_states = glob.glob(f"{exp_dir}/models/**/trainer_state.json")

    language_metrics = []
    # language_predictions = {}
    # language_label_ids = {}
    for language in tqdm(language_subset):
        for file in trainer_states:
            with open(file) as f:
                trainer_state = json.load(f)
            predictions, label_ids, metrics = evaluate_model_on_language(
                trainer_state["best_model_checkpoint"], language
            )
            language_metrics.append(metrics)
            # language_predictions[language] = predictions.tolist()
            # language_label_ids[language] = label_ids.tolist()

    language_metrics_df = pd.DataFrame(language_metrics)
    language_metrics_df.rename(
        columns={
            "test_loss": "loss",
            "test_accuracy": "accuracy",
            "test_f1_weighted": "f1",
            "test_precision_weighted": "precision",
            "test_recall_weighted": "recall",
            "test_runtime": "runtime",
            "test_samples_per_second": "samples_per_second",
        },
        inplace=True,
    )

    return language_metrics_df


# click cli command to evaluate an experiment
@click.command()
@click.argument("exp_name", type=str)
def main(exp_name: str):
    """Evaluate the experiment in the given directory."""
    exp_dir = f"nli_results/{exp_name}"
    language_metrics_df = evaluate_experiment(exp_dir)

    print(f"--- Saving the metrics in {exp_dir}/results/metrics.csv...")
    save_path = Path(f"{exp_dir}/results/metrics.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    language_metrics_df.to_csv(save_path, index=False)
