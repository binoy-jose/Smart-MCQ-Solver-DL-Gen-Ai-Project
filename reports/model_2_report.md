# Model 2 Report

## Objective

The objective of this model was to improve the baseline by using semantic sentence embeddings instead of TF-IDF features.

## Dataset

The same Smart MCQ Solver dataset was used, consisting of:

* Question prompt
* Five answer options (A, B, C, D, E)
* Correct answer label (training set)

The objective is to predict the three most probable answers for each question.

## Tasks Completed

* Loaded the pretrained MiniLM Sentence Transformer.
* Generated sentence embeddings for the question and answer options.
* Calculated cosine similarity between embeddings.
* Ranked the answer options.
* Generated Top-3 predictions.
* Evaluated the model using MAP@3.

## Methods Used

* Python
* Sentence Transformers
* MiniLM (all-MiniLM-L6-v2)
* Cosine Similarity

## Evaluation Metric

The model was evaluated using Mean Average Precision at 3 (MAP@3).

## Model Performance

**Model:** MiniLM + Cosine Similarity

* Training MAP@3: **0.4231**
* Kaggle Score: **0.38653**

**Remarks:** The MiniLM model improved the baseline by using contextual sentence embeddings.

## Outcome

The MiniLM model captured the meaning of the text better than TF-IDF and achieved a higher Kaggle score. It showed the advantage of using pretrained language models for semantic understanding.
