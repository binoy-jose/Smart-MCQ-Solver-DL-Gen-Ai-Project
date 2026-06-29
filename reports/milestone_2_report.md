# Milestone 2 Report

## Objective

The objective of Milestone 2 was to learn modern Natural Language Processing (NLP) techniques using transformer models and improve the baseline model developed in Milestone 1.

## Dataset

The same Smart MCQ Solver dataset was used, consisting of:

* Question prompt
* Five answer options (A, B, C, D, E)
* Correct answer label (training set)

The objective is to predict the three most probable answers for each question.

## Tasks Completed

* Loaded the dataset using the Hugging Face Datasets library.
* Explored the BERT tokenizer and vocabulary.
* Learned about special tokens (`[CLS]` and `[SEP]`).
* Tokenized the dataset using the BERT tokenizer.
* Extracted contextual embeddings using the BERT model.
* Explored the attention mechanism.
* Generated sentence embeddings using MiniLM.
* Calculated cosine similarity between prompts and answer options.
* Explored zero-shot classification using BART.
* Explored text generation using FLAN-T5.
* Built a MiniLM + Cosine Similarity model.
* Evaluated the model using MAP@3.

## Methods Used

* Python
* Hugging Face Datasets
* Hugging Face Transformers
* Sentence Transformers (MiniLM)
* BERT
* Cosine Similarity
* PyTorch

## Evaluation Metric

The model was evaluated using Mean Average Precision at 3 (MAP@3), which measures how highly the correct answer is ranked among the top three predictions.

## Model Performance

**Model:** Sentence Transformer (all-MiniLM-L6-v2) + Cosine Similarity

* Training MAP@3: **0.4231**
* Kaggle Score: **0.38653**

**Remarks:** The MiniLM model improved the baseline by using contextual sentence embeddings instead of TF-IDF features.

## Outcome

Milestone 2 introduced transformer-based NLP concepts and improved the baseline model using semantic sentence embeddings. The MiniLM model achieved better performance than the TF-IDF baseline and provided a strong foundation for developing more advanced deep learning models in the next milestones.
