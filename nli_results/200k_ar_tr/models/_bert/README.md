---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: _bert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# _bert

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/200k_ar_tr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8568
- Accuracy: 0.6161
- F1 Weighted: 0.6155
- Precision Weighted: 0.6333
- Recall Weighted: 0.6161

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
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.9165        | 1.0   | 6250  | 0.8988          | 0.5715   | 0.5699      | 0.6020             | 0.5715          |
| 0.8368        | 2.0   | 12500 | 0.9084          | 0.5743   | 0.5687      | 0.6315             | 0.5743          |
| 0.7548        | 3.0   | 18750 | 0.8568          | 0.6161   | 0.6155      | 0.6333             | 0.6161          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
