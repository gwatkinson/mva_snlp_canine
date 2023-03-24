from typing import Any

import numpy as np
import torch
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mva_snlp_canine.nli.dataset import tokenize_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    scores = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_weighted": precision_score(labels, predictions, average="weighted"),
        "recall_weighted": recall_score(labels, predictions, average="weighted"),
    }

    return scores


def finetune_model(
    model_name_or_path: str,
    dataset: Any,
    dataset_is_tokenized: bool,
    num_labels: int,
    training_kwargs: dict,
    output_dir: str,
    save_local: bool,
    hub_path: str,
    push_to_hub: bool,
    token: str,
):
    if not dataset_is_tokenized:
        print("--- Tokenizing the dataset...")
        dataset = tokenize_dataset(
            dataset=dataset,
            model_name_or_path=model_name_or_path,
            save_path=None,
            hub_path=None,
            n_jobs=12,
            no_pbar=False,
            push_to_hub=False,
            token=None,
        )

    print(f"--- Loading the model {model_name_or_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    print("--- Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=4)

    print(f"--- Cuda is available: {torch.cuda.is_available()}")

    print("--- Preparing the training...")
    if push_to_hub:
        exp_name = hub_path
    else:
        exp_name = output_dir

    training_args = TrainingArguments(
        output_dir=exp_name,
        push_to_hub=push_to_hub,
        **training_kwargs,
    )

    if push_to_hub:
        login(token=token)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"--- Training the model, pushing to {hub_path}...")
    trainer.train()

    if save_local:
        print("--- Saving the model locally...")
        trainer.save_state()
        trainer.save_model(output_dir)

    if push_to_hub:
        print("--- Pushing the model to the hub...")
        trainer.push_to_hub()
