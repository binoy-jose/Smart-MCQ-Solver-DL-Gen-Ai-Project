# Milestone 5 Report

## Objective

The objective of Milestone 5 was to learn how to improve prediction performance without retraining the model. This milestone focused on using **Softmax probabilities**, **model ensembling**, and **Test-Time Augmentation (TTA)** to generate more accurate answer rankings for the Smart MCQ Solver Challenge.

---

## Dataset

The same **Smart MCQ Solver** dataset was used, consisting of:

* Question prompt
* Five answer options (A, B, C, D, E)
* Correct answer label (training/validation set)
* Test set without answer labels

The milestone mainly focused on the **inference stage**, where trained transformer models were used to predict the most likely answer options.

---

## Tasks Completed

* Loaded pretrained transformer sequence classification models:

  * DeBERTa (`microsoft/deberta-v3-small`)
  * RoBERTa (`roberta-base`)
* Performed inference on individual questions.
* Applied the Softmax function to convert logits into class probabilities.
* Compared predictions from both models.
* Implemented simple probability averaging for model ensembling.
* Implemented weighted probability averaging using:

  * DeBERTa weight = 0.70
  * RoBERTa weight = 0.30
* Ranked answer options using the ensemble probabilities.
* Generated Top-3 predictions in the required Kaggle submission format.
* Created a `submission.csv` file for competition submission.
* Applied Test-Time Augmentation (TTA) by creating an instruction-augmented version of each prompt.
* Averaged probabilities from multiple inference passes.
* Compared predictions before and after TTA.
* Calculated MAP@3 on validation data.

---

## Methods Used

* Python
* PyTorch
* Hugging Face Transformers
* DeBERTa (`microsoft/deberta-v3-small`)
* RoBERTa (`roberta-base`)
* Softmax
* Probability Ensembling
* Weighted Ensembling
* Test-Time Augmentation (TTA)
* MAP@3 Evaluation

---

## Inference Pipeline

The pipeline used during this milestone consisted of the following steps:

1. Load the pretrained DeBERTa and RoBERTa models.
2. Tokenize the question and answer options.
3. Perform inference using each model.
4. Apply Softmax to convert logits into probabilities.
5. Combine probabilities using weighted averaging.
6. Rank all five answer options.
7. Select the Top-3 answers.
8. Generate predictions in Kaggle submission format.
9. Apply Test-Time Augmentation by creating an instruction-augmented prompt.
10. Average the probabilities from both inference passes.
11. Evaluate predictions using MAP@3 on the validation dataset.

---

## Evaluation Metrics

The following evaluation methods were explored during this milestone:

* Softmax Probability
* Confidence Score
* Top-1 Prediction
* Top-3 Prediction
* Mean Average Precision at 3 (MAP@3)

---

## Model Performance

### Models Used

**DeBERTa**

* Used as the primary transformer model.
* Generated probability scores for all five answer options.

**RoBERTa**

* Used as the second transformer model.
* Generated an independent set of probability scores.

---

### Ensemble Method

Two ensemble methods were explored.

**Simple Probability Averaging**

* Average of DeBERTa and RoBERTa probabilities.

**Weighted Probability Averaging**

* DeBERTa weight: **0.70**
* RoBERTa weight: **0.30**

The weighted probabilities were used to rank the answer options.

---

### Test-Time Augmentation (TTA)

A second version of each question was created by adding the instruction:

```
Answer the following multiple-choice question carefully:
```

Inference was performed on both the original and augmented prompts.

The final prediction was obtained by averaging the probabilities from both inference passes.

---

### Submission Generation

The final weighted ensemble predictions were converted into the required Kaggle submission format:

```
id,prediction
1,A C B
2,D A E
...
```

Each prediction contained the Top-3 ranked answer options.

---

## Key Observations

* Softmax converts model logits into probability scores that are easier to interpret.
* Model ensembling combines the strengths of multiple models and can improve prediction quality.
* Weighted ensembling allows stronger models to have greater influence on the final prediction.
* Test-Time Augmentation improves prediction stability by using multiple versions of the same input.
* Ranking the Top-3 answers is more suitable than predicting only one answer because the competition uses MAP@3 as the evaluation metric.
* MAP@3 rewards models that rank the correct answer as high as possible.
* Generating predictions in the correct Kaggle submission format is an important part of the inference pipeline.

---

## Outcome

Milestone 5 focused on improving inference performance rather than training new models. By combining predictions from DeBERTa and RoBERTa using probability ensembling, applying Test-Time Augmentation, and evaluating the results using MAP@3, the project demonstrated practical techniques for improving answer ranking. These methods can increase the robustness of transformer-based question-answering systems and provide better performance in competition-style evaluation.
