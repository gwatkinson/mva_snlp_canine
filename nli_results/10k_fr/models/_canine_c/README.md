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
- Loss: 1.2481
- Accuracy: 0.5888
- F1 Weighted: 0.5855
- Precision Weighted: 0.6212
- Recall Weighted: 0.5888

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
| 0.9635        | 1.0   | 312  | 1.0127          | 0.5241   | 0.5203      | 0.5650             | 0.5241          |
| 0.8574        | 2.0   | 625  | 0.9281          | 0.5843   | 0.5821      | 0.6237             | 0.5843          |
| 0.705         | 3.0   | 937  | 0.9454          | 0.6012   | 0.6009      | 0.6273             | 0.6012          |
| 0.5412        | 4.0   | 1250 | 1.0619          | 0.5924   | 0.5910      | 0.6169             | 0.5924          |
| 0.3842        | 4.99  | 1560 | 1.2481          | 0.5888   | 0.5855      | 0.6212             | 0.5888          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
