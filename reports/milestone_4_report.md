# Milestone 4 Report

## Objective

The objective of Milestone 4 was to formulate the Smart MCQ Solver Challenge as a **multiple-choice classification problem** using Hugging Face Transformers. The milestone focused on preparing the dataset for multiple-choice learning, fine-tuning a pretrained BERT model using **LoRA (Low-Rank Adaptation)**, and performing efficient inference by predicting probabilities for the five answer choices.

---

## Dataset

The same **Smart MCQ Solver** dataset was used, consisting of:

- Question prompt
- Five answer options (A, B, C, D, E)
- Correct answer label (training set)

Unlike previous milestones, each question was converted into **five prompt-option pairs**, allowing the model to evaluate all answer choices simultaneously as a multiple-choice classification task.

---

## Tasks Completed

- Converted answer labels (A–E) into numeric labels (0–4).
- Created prompt-option pairs using the format:
  ```
  prompt [SEP] option
  ```
- Tokenized each question into five separate input sequences using the BERT tokenizer.
- Reshaped the tokenized inputs into the format required by `AutoModelForMultipleChoice`.
- Loaded the pretrained **bert-base-uncased** multiple-choice model.
- Performed supervised forward passes using encoded labels to compute classification loss.
- Applied **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
- Created a Hugging Face Dataset containing tokenized inputs and labels.
- Performed a small LoRA fine-tuning experiment using Hugging Face Trainer.
- Generated answer probabilities using Softmax after fine-tuning.

---

## Methods Used

- Python
- Hugging Face Transformers
- BERT Tokenizer
- bert-base-uncased
- AutoModelForMultipleChoice
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- Hugging Face Datasets
- Hugging Face Trainer
- PyTorch
- Softmax

---

## Multiple-Choice Classification Pipeline

The pipeline implemented during this milestone consisted of the following stages:

1. Encode answer labels (A–E) into numeric values.
2. Create five prompt-option pairs for every question.
3. Tokenize all five choices using the BERT tokenizer.
4. Organize inputs into the format:
   ```
   Batch Size × Number of Choices × Sequence Length
   ```
5. Load the pretrained `bert-base-uncased` multiple-choice model.
6. Apply LoRA adapters to the attention layers.
7. Fine-tune only the LoRA parameters while freezing the original BERT weights.
8. Generate logits for the five answer choices.
9. Apply Softmax to convert logits into answer probabilities.
10. Select the option with the highest probability as the prediction.

---

## Evaluation Metrics

The following concepts were explored during this milestone:

- Cross-Entropy Loss for supervised multiple-choice learning.
- Softmax probabilities for converting logits into confidence scores.
- Global training steps using Hugging Face Trainer.
- Parameter efficiency using LoRA adapters.

---

## Model Performance

### Base Model

**bert-base-uncased**

- Used as the pretrained transformer backbone.
- Generated one logit for each of the five answer choices.

### Fine-Tuning Method

**LoRA (Low-Rank Adaptation)**

- Fine-tuned only a small subset of trainable adapter parameters.
- Reduced computational cost compared to full model fine-tuning.
- Preserved the pretrained knowledge of the original BERT model.

### Training Framework

**Hugging Face Trainer**

Training configuration included:

- Maximum sequence length: 64
- Batch size: 4
- Gradient accumulation steps: 1
- Maximum training steps: 4

A small fine-tuning experiment was performed to demonstrate the complete training workflow.

### Inference

The fine-tuned LoRA model generated logits for the five answer options.

Softmax was then applied to obtain probability scores for:

- Option A
- Option B
- Option C
- Option D
- Option E

The answer with the highest probability was selected as the final prediction.

---

## Key Observations

- Multiple-choice formatting allows the model to compare all answer options simultaneously.
- BERT produces one logit score for each available answer choice.
- Hugging Face's `AutoModelForMultipleChoice` simplifies multiple-choice classification by internally handling the comparison between choices.
- LoRA significantly reduces the number of trainable parameters while maintaining the pretrained model architecture.
- Fine-tuning with LoRA is more memory-efficient and computationally efficient than full model fine-tuning.
- Softmax converts raw logits into interpretable probability scores, enabling confidence-based answer ranking.
- Hugging Face Trainer provides a simple and efficient framework for training transformer-based multiple-choice models.

---

## Outcome

Milestone 4 introduced **parameter-efficient transformer fine-tuning** for the Smart MCQ Solver Challenge. By converting the dataset into the required multiple-choice format, applying LoRA to a pretrained BERT model, and training with Hugging Face Trainer, the project demonstrated an efficient workflow for transformer-based multiple-choice classification. The milestone also introduced probabilistic inference using Softmax, enabling the model to estimate confidence scores for each answer option. These techniques provide the foundation for developing larger transformer-based models and improving answer-ranking performance in future milestones.
