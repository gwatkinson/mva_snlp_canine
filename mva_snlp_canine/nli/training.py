import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def finetune_model(
    model_name_or_path: str,
    dataset_name_or_path: str,
    num_labels: int,
    output_dir: str,
    training_kwargs: dict,
    hub_path: str,
    token: str,
):
    print(f"--- Loading the dataset from {dataset_name_or_path}...")
    dataset = load_dataset(dataset_name_or_path)

    print(f"--- Loading the model {model_name_or_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    print("--- Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=4)

    print(f"--- Cuda is available: {torch.cuda.is_available()}")

    print("--- Preparing the training...")
    if hub_path:
        exp_name = hub_path
        push_to_hub = True
    else:
        exp_name = output_dir
        push_to_hub = False

    training_args = TrainingArguments(
        output_dir=exp_name,
        push_to_hub=push_to_hub,
        **training_kwargs,
    )

    # clf_metrics = evaluate.combine([
    #     evaluate.load("accuracy", average="weighted"),
    #     evaluate.load("f1", average="weighted"),
    #     evaluate.load("precision", average="weighted"),
    #     evaluate.load("recall", average="weighted"),
    # ])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        scores = {
            "accuracy": accuracy_score(labels, predictions),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
            "precision_weighted": precision_score(
                labels, predictions, average="weighted"
            ),
            "recall_weighted": recall_score(labels, predictions, average="weighted"),
        }

        return scores

    if hub_path:
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

    if hub_path:
        trainer.push_to_hub()
    else:
        trainer.save_state()
        trainer.save_model(exp_name)
