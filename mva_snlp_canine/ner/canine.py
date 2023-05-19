#Algorithms for speech and language processing
#Project 4. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation
#Task NER, Named Entity Recognition
#CANINE, model with no fast tokenizer 
#Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/token_classification.ipynb

import click
import numpy as np
from datasets import load_dataset, DatasetDict, load_metric
from transformers import CanineTokenizer, CanineForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
#from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

################----------------

@click.command()
@click.option('--lang', 'language', help='Language of the dataset. Choose between: "en","es", "nl","amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"', required=True)
@click.option('--batch_size'      , help='Batch size', default=4)
@click.option('--space_value'     , help='Value of " " (Unicode character 32)', default=0)
@click.option('--only_first'      , help='Controls whether if only the first subtoken of the word counts', default=False)

def run_script(language,batch_size,space_value,only_first):
    model_checkpoint = "google/canine-c"
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

    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer = CanineTokenizer.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
        tokens = examples["tokens"]
        labels = examples["ner_tags"]
                                
        #space_value=0
        #only_first=False
        next_values=-100
        sep_cls_value = -100

        opening_marks = ["(", "¡","¿"]
        closing_marks = ["!", ")", ".",":", ";", "?", ","]

        sentence = tokens[0]
        label_token = [sep_cls_value]+[labels[0]] * len(tokens[0])

        for i in range(1,len(tokens)):
            if tokens[i] not in closing_marks and tokens[i-1] not in opening_marks:
                sentence += " "
                label_token += [space_value]
            sentence += tokens[i]
            label_token += [labels[i]]
            if only_first:
                label_token += ([next_values] * (len(tokens[i])-1))
            else:
                label_token += ([labels[i]] * (len(tokens[i])-1))

        label_token += [sep_cls_value]

        tokenized_inputs = tokenizer(sentence,
                                truncation=True)
        
        label_token = label_token[:2048]
        tokenized_inputs["labels"] = label_token

        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=False)

    #model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model = CanineForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

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

    data_collator = DataCollatorForTokenClassification(tokenizer, padding='max_length',max_length =2048)
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
    run_script()

'''
ADDENDUM: Our results for "en","es", "nl" (CoNLL) and "pcm" (MasakhaNER) are unexpectedly good. After 
checking the code, we found a possible reason why this happens. The webpage above, which received an 
update during the development of the project, just suggested considering just the first token of the word 
or all of them, while the new version directly does this without bringing it up.

For mBERT this may work, but for CANINE we thought more sensible taking into account all the characters 
including spaces, as these also stay when preprocessing with "[ord(c) for c in text]" (In fact, the 
authors' webpage https://github.com/google-research/language/tree/master/language/canine states that 
"you will get the best performance by feeding natural text---without any punctuation splitting, etc.---to 
CANINE.")

These are the reasons behind space_value, only_first and next_values variables in the function 
tokenize_and_align_labels. However, the dataset include this system of classes: 0 for 'O', odds for the 
beginner token of the named entity and the following even for intermediate token of the same class. In our 
implementation, ALL the characters from this beginner token are labelled as beginner, when it may be more 
appropiate to label this way just the first character and the following as intermediate. 

A way to implement the latter would be changing lines 65-68 for the following:
# ---
            if only_first:
                label_token += ([next_values] * (len(tokens[i])-1))
            elif labels[i]%2 == 1:
                label_token += ([labels[i]+1] * (len(tokens[i])-1))
            else:
                label_token += ([labels[i]] * (len(tokens[i])-1))
# ---
'''