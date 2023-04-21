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

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/200k_ar_tr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8975
- Accuracy: 0.6016
- F1 Weighted: 0.6011
- Precision Weighted: 0.6189
- Recall Weighted: 0.6016

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
| 0.9351        | 1.0   | 6250  | 0.9371          | 0.5566   | 0.5527      | 0.5953             | 0.5566          |
| 0.8617        | 2.0   | 12500 | 0.9031          | 0.5755   | 0.5727      | 0.6182             | 0.5755          |
| 0.7837        | 3.0   | 18750 | 0.8975          | 0.6016   | 0.6011      | 0.6189             | 0.6016          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
