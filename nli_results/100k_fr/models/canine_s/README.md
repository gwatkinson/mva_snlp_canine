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

This model is a fine-tuned version of [google/canine-s](https://huggingface.co/google/canine-s) on the Gwatk/100k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0761
- Accuracy: 0.6695
- F1 Weighted: 0.6689
- Precision Weighted: 0.6941
- Recall Weighted: 0.6695

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
| 0.7992        | 1.0   | 2083  | 0.7814          | 0.6542   | 0.6561      | 0.6809             | 0.6542          |
| 0.7249        | 2.0   | 4166  | 0.7714          | 0.6767   | 0.6780      | 0.6991             | 0.6767          |
| 0.5968        | 3.0   | 6250  | 0.7732          | 0.6819   | 0.6814      | 0.7053             | 0.6819          |
| 0.4422        | 4.0   | 8333  | 0.8946          | 0.6739   | 0.6734      | 0.6975             | 0.6739          |
| 0.3125        | 5.0   | 10415 | 1.0761          | 0.6695   | 0.6689      | 0.6941             | 0.6695          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
