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

This model is a fine-tuned version of [google/canine-s](https://huggingface.co/google/canine-s) on the Gwatk/300k_all_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8850
- Accuracy: 0.6438
- F1 Weighted: 0.6428
- Precision Weighted: 0.6582
- Recall Weighted: 0.6438

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
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.8844        | 1.0   | 9375  | 0.8927          | 0.5876   | 0.5865      | 0.6086             | 0.5876          |
| 0.8269        | 2.0   | 18750 | 0.8914          | 0.6016   | 0.5974      | 0.6498             | 0.6016          |
| 0.7535        | 3.0   | 28125 | 0.8443          | 0.6273   | 0.6263      | 0.6662             | 0.6273          |
| 0.6668        | 4.0   | 37500 | 0.8342          | 0.6369   | 0.6343      | 0.6627             | 0.6369          |
| 0.5424        | 5.0   | 46875 | 0.8850          | 0.6438   | 0.6428      | 0.6582             | 0.6438          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
