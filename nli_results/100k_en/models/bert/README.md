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

This model is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) on the Gwatk/100k_en_xnli_subset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9212
- Accuracy: 0.7847
- F1 Weighted: 0.7850
- Precision Weighted: 0.7905
- Recall Weighted: 0.7847

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
| 0.6253        | 1.0   | 2083  | 0.5877          | 0.7655   | 0.7658      | 0.7680             | 0.7655          |
| 0.5171        | 2.0   | 4166  | 0.5455          | 0.7831   | 0.7836      | 0.7852             | 0.7831          |
| 0.3468        | 3.0   | 6250  | 0.6378          | 0.7699   | 0.7705      | 0.7828             | 0.7699          |
| 0.1927        | 4.0   | 8333  | 0.7527          | 0.7831   | 0.7830      | 0.7864             | 0.7831          |
| 0.1189        | 5.0   | 10415 | 0.9212          | 0.7847   | 0.7850      | 0.7905             | 0.7847          |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.3
