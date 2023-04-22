---
language:
- multilingual
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: bert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/300k_all_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9200
- Accuracy: 0.6602
- F1 Weighted: 0.6604
- Precision Weighted: 0.6733
- Recall Weighted: 0.6602

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
| 0.8202        | 1.0   | 9375  | 0.8344          | 0.6325   | 0.6327      | 0.6549             | 0.6325          |
| 0.7666        | 2.0   | 18750 | 0.8395          | 0.6317   | 0.6307      | 0.6713             | 0.6317          |
| 0.6799        | 3.0   | 28125 | 0.7805          | 0.6703   | 0.6698      | 0.6823             | 0.6703          |
| 0.5542        | 4.0   | 37500 | 0.8052          | 0.6635   | 0.6633      | 0.6766             | 0.6635          |
| 0.4347        | 5.0   | 46875 | 0.9200          | 0.6602   | 0.6604      | 0.6733             | 0.6602          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
