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

This model is a fine-tuned version of [google/canine-c](https://huggingface.co/google/canine-c) on the Gwatk/100k_zh_th_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1423
- Accuracy: 0.5369
- F1 Weighted: 0.5325
- Precision Weighted: 0.5458
- Recall Weighted: 0.5369

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
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1 Weighted | Precision Weighted | Recall Weighted |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:-----------:|:------------------:|:---------------:|
| 0.9391        | 1.0   | 3125  | 0.9977          | 0.5088   | 0.5022      | 0.5204             | 0.5088          |
| 0.8885        | 2.0   | 6250  | 0.9366          | 0.5578   | 0.5583      | 0.5641             | 0.5578          |
| 0.8027        | 3.0   | 9375  | 0.9969          | 0.5438   | 0.5428      | 0.5534             | 0.5438          |
| 0.7037        | 4.0   | 12500 | 1.0156          | 0.5458   | 0.5426      | 0.5547             | 0.5458          |
| 0.5657        | 5.0   | 15625 | 1.1423          | 0.5369   | 0.5325      | 0.5458             | 0.5369          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
