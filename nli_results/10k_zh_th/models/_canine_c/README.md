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

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/10k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0889
- Accuracy: 0.3972
- F1 Weighted: 0.3136
- Precision Weighted: 0.4038
- Recall Weighted: 0.3972

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
| 1.1022        | 1.0   | 312  | 1.1005          | 0.3333   | 0.1667      | 0.1111             | 0.3333          |
| 1.0988        | 2.0   | 625  | 1.0955          | 0.3775   | 0.2720      | 0.2903             | 0.3775          |
| 1.1002        | 3.0   | 937  | 1.0949          | 0.3807   | 0.3021      | 0.2545             | 0.3807          |
| 1.0618        | 4.0   | 1250 | 1.1000          | 0.3771   | 0.2621      | 0.3483             | 0.3771          |
| 1.0115        | 4.99  | 1560 | 1.0889          | 0.3972   | 0.3136      | 0.4038             | 0.3972          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
