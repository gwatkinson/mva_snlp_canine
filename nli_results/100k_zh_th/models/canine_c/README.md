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

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/100k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9264
- Accuracy: 0.5795
- F1 Weighted: 0.5800
- Precision Weighted: 0.5881
- Recall Weighted: 0.5795

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
| 0.9456        | 1.0   | 9375  | 0.9846          | 0.5253   | 0.5209      | 0.5607             | 0.5253          |
| 0.867         | 2.0   | 18750 | 0.9402          | 0.5651   | 0.5630      | 0.5761             | 0.5651          |
| 0.8087        | 3.0   | 28125 | 0.9109          | 0.5855   | 0.5852      | 0.6016             | 0.5855          |
| 0.7029        | 4.0   | 37500 | 0.9264          | 0.5795   | 0.5800      | 0.5881             | 0.5795          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
