---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: canine_s
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# canine_s

This model is a fine-tuned version of [google/canine-s](https://huggingface.co/google/canine-s) on the Gwatk/100k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8881
- Accuracy: 0.6112
- F1 Weighted: 0.6101
- Precision Weighted: 0.6204
- Recall Weighted: 0.6112

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
| 0.9203        | 1.0   | 9375  | 0.9568          | 0.5418   | 0.5368      | 0.5590             | 0.5418          |
| 0.8561        | 2.0   | 18750 | 0.8984          | 0.5944   | 0.5906      | 0.6096             | 0.5944          |
| 0.7888        | 3.0   | 28125 | 0.8807          | 0.6060   | 0.6046      | 0.6163             | 0.6060          |
| 0.6807        | 4.0   | 37500 | 0.8881          | 0.6112   | 0.6101      | 0.6204             | 0.6112          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
