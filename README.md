# DL-GenAi-Project-t2-26
A Deep Learning and Generative AI project that builds intelligent models to predict the **Top-3 correct answers** for challenging multiple-choice questions.

## Project Overview

This project was developed as part of the **Deep Learning & Generative AI (T2 2026)** course at IIT Madras.

The objective is to build and compare different machine learning and deep learning models for the **Smart MCQ Solver Challenge**. The models are evaluated using the **MAP@3 (Mean Average Precision @ 3)** metric.

## Models Implemented

- **Model 1:** TF-IDF + Cosine Similarity
- **Model 2:** MiniLM Sentence Embeddings + Cosine Similarity
- **Model 3:** Multi-Layer Perceptron (MLP) using TF-IDF Features
- **Model 4:** BGE Embeddings + Multi-Layer Perceptron (MLP)
- **Model 5:** RoBERTa Fine-Tuning
- **Model 6:** DeBERTa Fine-Tuning

## Live Application

The best model is deployed

https://quirk-mcq-407847884133.asia-south1.run.app

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- Weights & Biases (W&B)
- NumPy
- Pandas

## Evaluation Metric

The primary evaluation metric used in this project is:

- **MAP@3 (Mean Average Precision @ 3)**

Additional metrics include:

- Accuracy
- Precision
- Recall
- F1 Score

## Author

**Binoy Jose J C**

Student ID: **25ds1000066**
