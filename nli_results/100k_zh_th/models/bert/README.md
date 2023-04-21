---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: bert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/100k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8109
- Accuracy: 0.6542
- F1 Weighted: 0.6538
- Precision Weighted: 0.6673
- Recall Weighted: 0.6542

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.8839        | 1.0   | 9375  | 0.8924          | 0.5904   | 0.5905      | 0.5920             | 0.5904          |
| 0.8015        | 2.0   | 18750 | 0.8714          | 0.6112   | 0.6070      | 0.6541             | 0.6112          |
| 0.7363        | 3.0   | 28125 | 0.8244          | 0.6446   | 0.6453      | 0.6599             | 0.6446          |
| 0.6371        | 4.0   | 37500 | 0.8109          | 0.6542   | 0.6538      | 0.6673             | 0.6542          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
