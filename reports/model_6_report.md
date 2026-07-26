# Model 6 Report

## Objective

The objective of this model was to evaluate whether a more advanced pretrained transformer model could further improve the performance of the Smart MCQ Solver Challenge. A pretrained **DeBERTa** model was fine-tuned using the Hugging Face Transformers library and compared with the RoBERTa model. DeBERTa uses an improved attention mechanism that helps the model better understand the relationship between the question and each answer option.

## Dataset

The same Smart MCQ Solver dataset was used.

Each question was converted into five question-option pairs. The correct option was assigned a label of **1**, while the remaining four options were assigned a label of **0**. This converted the multiple-choice problem into a binary classification task.

## Tasks Completed

- Converted the dataset into question-option pairs.
- Split the dataset into training and validation sets.
- Tokenized the Question–Option pairs using the pretrained DeBERTa tokenizer.
- Converted the data into Hugging Face Dataset format.
- Fine-tuned the pretrained DeBERTa model.
- Used the Hugging Face Trainer API for model training.
- Applied Early Stopping to prevent overfitting.
- Evaluated the model using the validation dataset.
- Generated Top-3 predictions.
- Calculated MAP@3.
- Saved the best-performing model and tokenizer.
- Tracked the experiment using Weights & Biases (W&B).

## Methods Used

- Python
- PyTorch
- Hugging Face Transformers
- DeBERTa
- Hugging Face Trainer
- AdamW Optimizer
- Early Stopping
- Weights & Biases (W&B)

## Evaluation Metric

The model was evaluated using Validation Accuracy, Validation F1 Score, and Mean Average Precision at 3 (MAP@3).

## Model Performance

**Model:** DeBERTa

- Validation Accuracy: **98.00%**
- Validation F1 Score: **97.90%**
- Validation MAP@3: **0.98625**
- Kaggle Score: **0.75602**

**Remarks:** DeBERTa achieved better performance than the RoBERTa model on both the validation dataset and the Kaggle leaderboard. Its improved attention mechanism enabled the model to capture richer contextual information, resulting in better generalization on unseen questions.

## Outcome

This model successfully fine-tuned the pretrained DeBERTa transformer for the Smart MCQ Solver Challenge. It demonstrated the effectiveness of advanced transformer architectures for multiple-choice question answering. Among all the models developed in this project, DeBERTa achieved the best Kaggle performance and produced the strongest overall results.
