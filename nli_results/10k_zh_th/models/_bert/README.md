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

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/10k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3869
- Accuracy: 0.5932
- F1 Weighted: 0.5903
- Precision Weighted: 0.6218
- Recall Weighted: 0.5932

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
| 0.9378        | 1.0   | 312  | 1.0253          | 0.4651   | 0.4216      | 0.6050             | 0.4651          |
| 0.8566        | 2.0   | 625  | 0.9122          | 0.5920   | 0.5928      | 0.6102             | 0.5920          |
| 0.6732        | 3.0   | 937  | 0.9396          | 0.6012   | 0.6008      | 0.6287             | 0.6012          |
| 0.4866        | 4.0   | 1250 | 1.1985          | 0.5867   | 0.5812      | 0.6188             | 0.5867          |
| 0.3169        | 4.99  | 1560 | 1.3869          | 0.5932   | 0.5903      | 0.6218             | 0.5932          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
