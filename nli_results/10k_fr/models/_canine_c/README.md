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

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/10k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0983
- Accuracy: 0.3333
- F1 Weighted: 0.1667
- Precision Weighted: 0.1111
- Recall Weighted: 0.3333

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.01
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 1.1009        | 1.0   | 312  | 1.1016          | 0.3333   | 0.1667      | 0.1111             | 0.3333          |
| 1.1002        | 2.0   | 625  | 1.0993          | 0.3333   | 0.1667      | 0.1111             | 0.3333          |
| 1.1009        | 3.0   | 937  | 1.0990          | 0.3333   | 0.1667      | 0.1111             | 0.3333          |
| 1.0988        | 4.0   | 1250 | 1.0986          | 0.3333   | 0.1667      | 0.1111             | 0.3333          |
| 1.1           | 4.99  | 1560 | 1.0983          | 0.3333   | 0.1667      | 0.1111             | 0.3333          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
