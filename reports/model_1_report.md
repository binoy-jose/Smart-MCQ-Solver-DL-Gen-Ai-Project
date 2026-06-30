# Model 1 Report

## Objective

The objective of this model was to build a simple baseline for the Smart MCQ Solver Challenge using TF-IDF features and cosine similarity.

## Dataset

The dataset consists of:

* Question prompt
* Five answer options (A, B, C, D, E)
* Correct answer label (training set)

The objective is to predict the three most probable answers for each question.

## Tasks Completed

* Cleaned the text data.
* Generated TF-IDF features.
* Calculated cosine similarity between the prompt and each answer option.
* Ranked the answer options based on similarity.
* Generated Top-3 predictions.
* Evaluated the model using MAP@3.

## Methods Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

## Evaluation Metric

The model was evaluated using Mean Average Precision at 3 (MAP@3), which measures how highly the correct answer is ranked among the top three predictions.

## Model Performance

**Model:** TF-IDF + Cosine Similarity

* Training MAP@3: **0.2962**
* Kaggle Score: **0.30922**

**Remarks:** This model serves as the baseline for comparing future deep learning models.

## Outcome

The TF-IDF baseline provided a simple and effective starting point for the project. It helped in understanding text preprocessing, feature extraction, cosine similarity, and the MAP@3 evaluation metric.
