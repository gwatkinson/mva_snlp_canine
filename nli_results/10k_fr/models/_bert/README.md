---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: _bert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# _bert

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/10k_fr_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 1.4166
- Accuracy: 0.6382
- F1 Weighted: 0.6371
- Precision Weighted: 0.6584
- Recall Weighted: 0.6382

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
| 0.8671        | 1.0   | 312  | 0.9368          | 0.5313   | 0.5097      | 0.6534             | 0.5313          |
| 0.7168        | 2.0   | 625  | 0.8113          | 0.6394   | 0.6379      | 0.6481             | 0.6394          |
| 0.4484        | 3.0   | 937  | 0.9821          | 0.6414   | 0.6378      | 0.6737             | 0.6414          |
| 0.2703        | 4.0   | 1250 | 1.2011          | 0.6482   | 0.6474      | 0.6709             | 0.6482          |
| 0.1571        | 4.99  | 1560 | 1.4166          | 0.6382   | 0.6371      | 0.6584             | 0.6382          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
