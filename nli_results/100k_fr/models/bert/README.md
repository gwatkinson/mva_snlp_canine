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

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/100k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0296
- Accuracy: 0.7305
- F1 Weighted: 0.7305
- Precision Weighted: 0.7437
- Recall Weighted: 0.7305

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
| 0.7105        | 1.0   | 2083  | 0.6743          | 0.7289   | 0.7297      | 0.7358             | 0.7289          |
| 0.6087        | 2.0   | 4166  | 0.6521          | 0.7281   | 0.7289      | 0.7312             | 0.7281          |
| 0.4395        | 3.0   | 6250  | 0.7516          | 0.7249   | 0.7260      | 0.7430             | 0.7249          |
| 0.2717        | 4.0   | 8333  | 0.8963          | 0.7153   | 0.7148      | 0.7367             | 0.7153          |
| 0.1617        | 5.0   | 10415 | 1.0296          | 0.7305   | 0.7305      | 0.7437             | 0.7305          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
