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

This model is a fine-tuned version of [google/canine-s](https://huggingface.co/google/canine-s) on the Gwatk/10k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1543
- Accuracy: 0.5859
- F1 Weighted: 0.5860
- Precision Weighted: 0.6032
- Recall Weighted: 0.5859

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
| 0.9798        | 1.0   | 312  | 1.0487          | 0.4635   | 0.4365      | 0.5442             | 0.4635          |
| 0.8988        | 2.0   | 625  | 0.9337          | 0.5699   | 0.5680      | 0.5976             | 0.5699          |
| 0.7632        | 3.0   | 937  | 0.9914          | 0.5779   | 0.5776      | 0.6119             | 0.5779          |
| 0.6315        | 4.0   | 1250 | 1.0542          | 0.5839   | 0.5837      | 0.6008             | 0.5839          |
| 0.5297        | 4.99  | 1560 | 1.1543          | 0.5859   | 0.5860      | 0.6032             | 0.5859          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
