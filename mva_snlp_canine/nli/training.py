import evaluate
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


def finetune_model(
    model_name_or_path: str,
    dataset_name_or_path: str,
    num_labels: int,
    output_dir: str,
    hub_path: str,
):
    print(f"--- Loading the dataset from {dataset_name_or_path}...")
    dataset = load_dataset(dataset_name_or_path)

    print(f"--- Loading the model {model_name_or_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    print("--- Preparing the training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return clf_metrics.compute(predictions=predictions, references=labels)

    if hub_path:
        login()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        push_to_hub=hub_path,
    )

    print(f"--- Training the model, pushing to {hub_path}...")
    trainer.train()
