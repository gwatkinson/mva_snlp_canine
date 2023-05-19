#Algorithms for speech and language processing
#Project 4. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation
#Task NER, Named Entity Recognition
#mBERT (benchmark), model with fast tokenizer
#Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/token_classification.ipynb

import click
import numpy as np
from datasets import load_dataset, DatasetDict, load_metric
#from transformers import CanineTokenizer, CanineForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

################----------------

@click.command()
@click.option('--lang', 'language', help='Language of the dataset. Choose between: "en","es", "nl","amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"', required=True)
@click.option('--batch_size'      , help='Batch size', default=16)

def run_script(language,batch_size):
    model_checkpoint = "bert-base-multilingual-cased"
    #batch_size = 1
    #lang = "es"

    if language == "en":
        datasets = load_dataset("conll2003")
    elif language in ["es", "nl"]:
        dataset_train = load_dataset("conll2002", language, split='train[:-1]')
        dataset_validation = load_dataset("conll2002", language, split='validation[:-1]')
        dataset_test = load_dataset("conll2002", language, split='test[:-1]')
        datasets = DatasetDict({"train":dataset_train, "validation":dataset_validation, "test":dataset_test})
    elif language in ['amh', 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol', 'yor']:
        datasets = load_dataset("masakhaner",language)
    else:
        print("Wrong language")

    label_list = datasets["train"].features[f"ner_tags"].feature.names
    print(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #tokenizer = CanineTokenizer.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    #model = CanineForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-ner-{language}",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    #trainer.evaluate(tokenized_datasets["test"])
    #print(trainer.evaluate(tokenized_datasets["test"]))

    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)

    print("\n\n","---RESULTS FOR THE TEST DATASET:--- \n")
    for i in results.keys():
        print(i,": \n",results[i],"\n")

#-----
if __name__ == "__main__":
    run_script() # pylint: disable=no-value-for-parameter