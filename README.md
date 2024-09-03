# my__model

This model is a fine-tuned version of [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) on the imagefolder dataset.
with specialised focus on kneeosteoarthritis data.
It achieves the following results on the evaluation set:
- Loss: 1.3439
- Accuracy: 0.4419

## Model description
model built to refine the classification with specialised focus on kneeosteoarthritis data.
for medical data related to similar domains can use the same to finetune further.




## results/inference example for a sample instance

[{'label': '1', 'score': 1.0},
 {'label': '0', 'score': 8.705698431673573e-39},
 {'label': '2', 'score': 0.0},
 {'label': '3', 'score': 0.0},
 {'label': '4', 'score': 0.0}]
predicted the given instance to be of 1
### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.3665        | 1.0   | 104  | 1.3439          | 0.4419   |


### Framework versions

- Transformers 4.42.4
- Pytorch 2.4.0+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1
