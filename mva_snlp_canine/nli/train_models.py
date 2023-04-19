from typing import Any

import click
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

from mva_snlp_canine.nli.tokenize_dataset import tokenize_dataset
from mva_snlp_canine.utils import load_config_nli, load_dataset_from_config


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
    dataset_name: str,
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
            save_local=False,
            push_to_hub=False,
            token=None,
        )

    print(f"\n--- Loading the model {model_name_or_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    print("\n--- Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=4)

    print(f"\n--- Cuda is available: {torch.cuda.is_available()}")

    print("\n--- Preparing the training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_path,
        hub_strategy="end",
        push_to_hub=push_to_hub,
        hub_token=token,
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

    print(f"\n--- Training the model, pushing to {hub_path}...")
    trainer.train()

    print("\n--- Evaluating the model...")
    predictions, label_ids, metrics = trainer.predict(dataset["test"])
    trainer.log_metrics("test", metrics)

    print("\n--- Creating the model card...")
    trainer.create_model_card(
        language="multilingual", finetuned_from=model_name_or_path, dataset=dataset_name
    )

    if save_local:
        print("\n--- Saving the model locally...")
        trainer.save_state()
        # trainer.save_model(output_dir)

    if push_to_hub:
        print("\n--- Pushing the model to the hub...")
        trainer.push_to_hub(blocking=False)


# CLI functions
@click.command()
@click.argument("config_file_path", type=str)
def main(config_file_path):
    cfg = load_config_nli(config_file_path)
    dataset_is_tokenized = cfg.DATASET_IS_TOKENISED

    model_list = cfg.MODEL_LIST
    model_postfix = cfg.MODEL_POSTFIX

    for model_name_or_path, postfix in zip(model_list, model_postfix):
        print(f"Training for {model_name_or_path}...")

        if cfg.DATASET_IS_TOKENISED:
            dataset = load_dataset_from_config(cfg, "tokenized", postfix=postfix)
        else:
            dataset = load_dataset_from_config(cfg, "preprocessed")

        experiment_output_dir = cfg.DIR_TEMPLATE_TRAINING.format(postfix=postfix)
        experiment_hub_path = cfg.HUB_TEMPLATE_TRAINING.format(postfix=postfix)

        finetune_model(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_is_tokenized=dataset_is_tokenized,
            num_labels=cfg.NUM_LABELS,
            training_kwargs=cfg.TRAINING_KWARGS,
            output_dir=experiment_output_dir,
            save_local=cfg.SAVE_LOCAL,
            hub_path=experiment_hub_path,
            push_to_hub=cfg.PUSH_TO_HUB,
            token=cfg.TOKEN,
            dataset_name=cfg.HUB_PATH_PREPROCESSED_DATASET,
        )


if __name__ == "__main__":
    main()
