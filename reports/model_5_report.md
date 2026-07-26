# Model 5 Report

## Objective

The objective of this model was to investigate whether fine-tuning a pretrained transformer model could improve the performance of the Smart MCQ Solver Challenge. A pretrained **RoBERTa-base** model was fine-tuned using the Hugging Face Transformers library to classify the correct answer from multiple-choice questions. The model learns contextual relationships between the question and each answer option instead of relying on manually engineered features. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Dataset

The same Smart MCQ Solver dataset was used.

Each question was converted into five question-option pairs. The correct option was assigned a label of **1**, while the remaining four options were assigned a label of **0**. This converted the multiple-choice problem into a binary classification task. :contentReference[oaicite:2]{index=2}

## Tasks Completed

- Converted the dataset into question-option pairs.
- Split the dataset into training and validation sets.
- Tokenized the Question–Option pairs using the pretrained RoBERTa tokenizer.
- Converted the data into Hugging Face Dataset format.
- Fine-tuned the pretrained RoBERTa-base model.
- Used the Hugging Face Trainer API for model training.
- Applied Early Stopping to prevent overfitting.
- Evaluated the model using the validation dataset.
- Generated Top-3 predictions.
- Calculated MAP@3.
- Saved the best-performing model and tokenizer.
- Tracked the experiment using Weights & Biases (W&B). :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

## Methods Used

- Python
- PyTorch
- Hugging Face Transformers
- RoBERTa-base
- Hugging Face Trainer
- AdamW Optimizer
- Early Stopping
- Weights & Biases (W&B)

## Evaluation Metric

The model was evaluated using Validation Accuracy, Validation F1 Score, and Mean Average Precision at 3 (MAP@3).

## Model Performance

**Model:** RoBERTa-base

- Validation Accuracy: **97.50%**
- Validation F1 Score: **97.43%**
- Validation MAP@3: **0.98250**
- Kaggle Score: **0.73233**

**Remarks:** Fine-tuning the pretrained RoBERTa model significantly improved the validation performance compared to the previous models. The transformer learned contextual information directly from the Question–Option pairs and achieved strong validation metrics. However, the Kaggle leaderboard score indicated that there was still room for improving generalization on unseen test questions.

## Outcome

This model successfully fine-tuned a pretrained RoBERTa transformer for the Smart MCQ Solver Challenge. It introduced transformer-based contextual learning, Hugging Face Trainer, early stopping, and pretrained language model fine-tuning into the project. The experiment demonstrated that transformer models can produce much stronger validation performance than traditional feature-based approaches and provided the foundation for evaluating more advanced transformer architectures in the next model.
