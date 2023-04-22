---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: canine_c
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# canine_c

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/300k_all_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9519
- Accuracy: 0.6502
- F1 Weighted: 0.6506
- Precision Weighted: 0.6652
- Recall Weighted: 0.6502

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
| 0.856         | 1.0   | 9375  | 0.8608          | 0.6112   | 0.6107      | 0.6244             | 0.6112          |
| 0.7679        | 2.0   | 18750 | 0.8543          | 0.6277   | 0.6260      | 0.6643             | 0.6277          |
| 0.6975        | 3.0   | 28125 | 0.8452          | 0.6546   | 0.6539      | 0.6859             | 0.6546          |
| 0.5763        | 4.0   | 37500 | 0.8424          | 0.6554   | 0.6562      | 0.6709             | 0.6554          |
| 0.4348        | 5.0   | 46875 | 0.9519          | 0.6502   | 0.6506      | 0.6652             | 0.6502          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
