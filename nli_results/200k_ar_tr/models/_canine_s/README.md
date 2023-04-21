---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: _canine_s
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# _canine_s

This model is a fine-tuned version of [google/canine-s](https://huggingface.co/google/canine-s) on the Gwatk/200k_ar_tr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9078
- Accuracy: 0.5904
- F1 Weighted: 0.5896
- Precision Weighted: 0.6095
- Recall Weighted: 0.5904

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
| 0.9624        | 1.0   | 6250  | 0.9557          | 0.5406   | 0.5384      | 0.5643             | 0.5406          |
| 0.8934        | 2.0   | 12500 | 0.9389          | 0.5538   | 0.5490      | 0.6117             | 0.5538          |
| 0.8218        | 3.0   | 18750 | 0.9078          | 0.5904   | 0.5896      | 0.6095             | 0.5904          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
