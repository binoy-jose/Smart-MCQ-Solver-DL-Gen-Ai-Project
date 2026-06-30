# Model 3 Report

## Objective

The objective of this model was to build the first deep learning model from scratch using PyTorch for the Smart MCQ Solver Challenge.

## Dataset

The same Smart MCQ Solver dataset was used.

Each question was converted into five question-option pairs. The correct option was given a label of **1**, and the remaining options were given a label of **0**.

## Tasks Completed

* Converted the dataset into question-option pairs.
* Generated TF-IDF features.
* Split the data into training and validation sets.
* Created custom PyTorch Dataset and DataLoader objects.
* Built a Multi-Layer Perceptron (MLP) from scratch.
* Trained the model using GPU.
* Evaluated the model on the validation dataset.
* Generated Top-3 predictions.
* Calculated MAP@3.
* Tracked the experiment using Weights & Biases (W&B).

## Methods Used

* Python
* PyTorch
* TF-IDF Vectorizer
* Multi-Layer Perceptron (MLP)
* Adam Optimizer
* CrossEntropy Loss
* Weights & Biases (W&B)

## Evaluation Metric

The model was evaluated using Validation Accuracy and Mean Average Precision at 3 (MAP@3).

## Model Performance

**Model:** MLP (From Scratch)

* Validation Accuracy: **96.60%**
* Training MAP@3: **0.9646**
* Kaggle Score: **0.72901**

**Remarks:** This was the first neural network built completely from scratch for the project and achieved a significant improvement over the previous models.

## Outcome

The MLP model successfully learned to classify question-option pairs and greatly improved the Kaggle score compared to the TF-IDF and MiniLM models. This model also introduced GPU training and experiment tracking using Weights & Biases (W&B), providing a strong foundation for developing more advanced transformer-based models.
