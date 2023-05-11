---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: canine_c
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# canine_c

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/100k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1151
- Accuracy: 0.6594
- F1 Weighted: 0.6583
- Precision Weighted: 0.6802
- Recall Weighted: 0.6594

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 6
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 48
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.7902        | 1.0   | 2083  | 0.7971          | 0.6671   | 0.6677      | 0.6809             | 0.6671          |
| 0.7135        | 2.0   | 4166  | 0.7462          | 0.6831   | 0.6844      | 0.6971             | 0.6831          |
| 0.5757        | 3.0   | 6250  | 0.8425          | 0.6639   | 0.6627      | 0.6946             | 0.6639          |
| 0.4181        | 4.0   | 8333  | 0.9972          | 0.6598   | 0.6576      | 0.6844             | 0.6598          |
| 0.293         | 5.0   | 10415 | 1.1151          | 0.6594   | 0.6583      | 0.6802             | 0.6594          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
