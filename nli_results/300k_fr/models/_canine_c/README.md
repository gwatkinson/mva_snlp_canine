---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: _canine_c
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# _canine_c

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/300k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7293
- Accuracy: 0.6904
- F1 Weighted: 0.6904
- Precision Weighted: 0.7101
- Recall Weighted: 0.6904

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
- num_epochs: 2
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.7669        | 1.0   | 9375  | 0.8232          | 0.6394   | 0.6386      | 0.6766             | 0.6394          |
| 0.6534        | 2.0   | 18750 | 0.7293          | 0.6904   | 0.6904      | 0.7101             | 0.6904          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
