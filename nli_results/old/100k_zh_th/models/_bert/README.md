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

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/100k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1088
- Accuracy: 0.6534
- F1 Weighted: 0.6538
- Precision Weighted: 0.6678
- Recall Weighted: 0.6534

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
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.8099        | 1.0   | 3125  | 0.8175          | 0.6410   | 0.6418      | 0.6598             | 0.6410          |
| 0.7083        | 2.0   | 6250  | 0.7790          | 0.6639   | 0.6649      | 0.6694             | 0.6639          |
| 0.5906        | 3.0   | 9375  | 0.8864          | 0.6610   | 0.6617      | 0.6880             | 0.6610          |
| 0.4421        | 4.0   | 12500 | 0.9257          | 0.6655   | 0.6657      | 0.6722             | 0.6655          |
| 0.3032        | 5.0   | 15625 | 1.1088          | 0.6534   | 0.6538      | 0.6678             | 0.6534          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
