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

This model is a fine-tuned version of [google/canine-s](https://huggingface.co/google/canine-s) on the Gwatk/100k_en_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9717
- Accuracy: 0.7189
- F1 Weighted: 0.7192
- Precision Weighted: 0.7367
- Recall Weighted: 0.7189

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
| 0.7593        | 1.0   | 2083  | 0.7003          | 0.7052   | 0.7057      | 0.7068             | 0.7052          |
| 0.6617        | 2.0   | 4166  | 0.6812          | 0.7205   | 0.7218      | 0.7383             | 0.7205          |
| 0.5041        | 3.0   | 6250  | 0.7160          | 0.7369   | 0.7379      | 0.7557             | 0.7369          |
| 0.3551        | 4.0   | 8333  | 0.8621          | 0.7141   | 0.7135      | 0.7353             | 0.7141          |
| 0.244         | 5.0   | 10415 | 0.9717          | 0.7189   | 0.7192      | 0.7367             | 0.7189          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
