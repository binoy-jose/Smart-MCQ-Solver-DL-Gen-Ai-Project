# Model 4 Report

## Objective

The objective of this model was to investigate whether replacing traditional TF-IDF features with pretrained semantic embeddings could improve the performance of the Smart MCQ Solver Challenge. A pretrained BGE (BAAI/bge-base-en-v1.5) embedding model was combined with a Multi-Layer Perceptron (MLP) classifier built using PyTorch.

## Dataset

The same Smart MCQ Solver dataset was used.

Each question was converted into five question-option pairs. The correct option was assigned a label of **1**, while the remaining four options were assigned a label of **0**.

## Tasks Completed

* Converted the dataset into question-option pairs.
* Generated semantic embeddings using the pretrained BGE (BAAI/bge-base-en-v1.5) model.
* Split the data into training and validation sets.
* Converted embeddings into PyTorch tensors.
* Created custom PyTorch Dataset and DataLoader objects.
* Built a Multi-Layer Perceptron (MLP) classifier.
* Trained the model using GPU acceleration.
* Applied Weighted CrossEntropy Loss to address class imbalance.
* Used the AdamW optimizer with a learning rate scheduler.
* Saved the best-performing model based on validation loss.
* Evaluated the model using the validation dataset.
* Generated Top-3 predictions.
* Calculated MAP@3.
* Tracked the experiment using Weights & Biases (W&B).

## Methods Used

* Python
* PyTorch
* Sentence Transformers
* BAAI/bge-base-en-v1.5
* Multi-Layer Perceptron (MLP)
* AdamW Optimizer
* Weighted CrossEntropy Loss
* ReduceLROnPlateau Learning Rate Scheduler
* Weights & Biases (W&B)

## Evaluation Metric

The model was evaluated using Validation Accuracy, Validation F1 Score, and Mean Average Precision at 3 (MAP@3).

## Model Performance

**Model:** BGE + MLP

* Validation Accuracy: **90.85%**
* Validation F1 Score: **79.55%**
* Validation MAP@3: **0.92917**
* Kaggle Score: **0.70074**

**Remarks:** Although the pretrained BGE embeddings produced strong validation performance, the model did not outperform the previous TF-IDF + MLP model on the Kaggle leaderboard. This experiment demonstrated that richer semantic embeddings alone were insufficient to improve generalization under the current pairwise classification framework.

## Outcome

This model successfully integrated pretrained transformer-based sentence embeddings with a custom-built neural network classifier. It introduced semantic feature representations, weighted loss functions, learning rate scheduling, best model checkpointing, and experiment tracking using Weights & Biases (W&B). While the public Kaggle score did not surpass Model 3, the experiment provided valuable insights into the effectiveness of pretrained embeddings for multiple-choice question ranking and established a strong baseline for exploring more advanced transformer-based architectures in subsequent models.
