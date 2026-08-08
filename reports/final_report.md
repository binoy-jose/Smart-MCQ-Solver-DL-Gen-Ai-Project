# Smart MCQ Solver Challenge - Final Project Report

**Course:** Introduction to Deep Learning and Generative AI  
**Platform:** Kaggle  
**Project:** Smart MCQ Solver Challenge  
**Student Name:** Binoy Jose J C      
**Roll Number:** 25DS1000066  

---

## 1. Abstract / Executive Summary

The Smart MCQ Solver Challenge is a multiple-choice question answering problem. Each question contains one prompt and five possible answer options labelled A, B, C, D, and E. The task is not only to identify one answer, but to rank the three most likely answers. The competition uses Mean Average Precision at 3 (MAP@3), which gives a higher score when the correct answer appears earlier in the predicted top three.

This project started with simple Natural Language Processing baselines and gradually moved toward deep learning and fine-tuned transformer models. The experiments included TF-IDF with cosine similarity, MiniLM sentence embeddings, a custom Multi-Layer Perceptron (MLP) using TF-IDF features, a tuned MLP, BGE embeddings with an MLP, fine-tuned RoBERTa-base, and fine-tuned DeBERTa-v3-small. All trainable ranking models used a pairwise formulation in which every question was combined separately with each of its five options.

The best model was **DeBERTa-v3-small**. It achieved a validation accuracy of **0.9900**, validation F1 score of **0.9893**, validation MAP@3 of **0.9950**, and a recorded Kaggle score of **0.75602**. The project showed that simple semantic similarity models were not sufficient for this task. Learning the relationship between a question and an answer option using supervised pairwise classification produced much better results. Fine-tuning a pretrained transformer gave the strongest final performance.

---

## 2. Introduction

### 2.1 Problem Statement

The objective of the Kaggle competition is to build a system that can solve multiple-choice questions. Every question contains:

- one question prompt,
- five answer choices: A, B, C, D, and E,
- one correct answer in the training data.

For the test data, the correct answer is hidden. The model must predict exactly three option labels in ranked order, for example:

```text
A C B
```

The first position is the most important because MAP@3 gives the highest reward when the correct answer is ranked first.

### 2.2 Project Objective

The main objectives of this project were to:

1. Understand the structure and characteristics of the MCQ dataset.
2. Create simple NLP baselines before moving to deep learning.
3. Learn how to use PyTorch for a custom neural network.
4. Experiment with sentence embedding models such as MiniLM and BGE.
5. Fine-tune pretrained transformer models using Hugging Face.
6. Use Weights & Biases (W&B) to track training and validation performance.
7. Improve the Kaggle MAP@3 score through systematic experimentation.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset Description

The final notebook used the Kaggle competition files `train.csv`, `test.csv`, and `sample_submission.csv`.

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| Training set | 2,000 | 8 | Model development and validation |
| Test set | 500 | 7 | Final Kaggle predictions |
| Sample submission | 500 | 2 | Required submission format |

The training columns were:

```text
id, prompt, A, B, C, D, E, answer
```

The test set contained the same fields except for the `answer` column.

### 3.2 Exploratory Data Analysis

#### Missing Values

No missing values were found in the 2,000 training questions. Therefore, no data imputation was required.

#### Duplicate Rows

The notebook found **0 duplicate rows**. This was useful because repeated questions could have biased the models toward duplicated examples.

#### Correct Answer Distribution

The correct answer labels were distributed as follows:

| Answer | Count |
|---|---:|
| A | 369 |
| B | 490 |
| C | 459 |
| D | 358 |
| E | 324 |

The labels were reasonably balanced, although B and C appeared more often than D and E.

#### Prompt Length

The average prompt length was approximately **117.67 characters** and **18.15 words**. The shortest prompt contained 3 words and the longest contained 51 words. This showed that most questions were short enough to fit easily inside transformer sequence limits.

#### Option Length

The answer choices had similar average lengths. Each option contained about **26 words on average**. Therefore, option length alone was unlikely to be a reliable signal for the correct answer.

### 3.3 Text Preprocessing

Different models required different preprocessing strategies.

For the TF-IDF models, the text was:

- converted to lowercase,
- stripped of punctuation,
- cleaned by replacing repeated whitespace with a single space.

For pretrained transformer models, aggressive cleaning was avoided. The original question and answer text were passed to the pretrained tokenizer because punctuation, word form, and sentence structure can contain useful information.

### 3.4 Pairwise Question-Option Transformation

A major design decision was to transform each MCQ into five question-option pairs.

For example:

```text
Original question
    |
    +-- Question + Option A -> label 0 or 1
    +-- Question + Option B -> label 0 or 1
    +-- Question + Option C -> label 0 or 1
    +-- Question + Option D -> label 0 or 1
    +-- Question + Option E -> label 0 or 1
```

The correct option received label `1`, while the other four options received label `0`. After prediction, the five positive-class probabilities were ranked and the highest three were returned.

With an 80:20 question-level split:

- 1,600 training questions became 8,000 question-option training pairs.
- 400 validation questions became 2,000 question-option validation pairs.

The split was performed at the **question level before creating pairs**. This was important because splitting after pair creation could place different options from the same question in both training and validation data, causing data leakage.

### 3.5 Data Augmentation

No synthetic text augmentation was used. The conversion from one MCQ into five question-option pairs was a **task reformulation**, not synthetic data augmentation. This approach increased the number of supervised training samples while keeping all content faithful to the original dataset.

---

## 4. Tokenization Strategy

### 4.1 Primary Tokenizer: DeBERTa-v3-small

The best-performing model was `microsoft/deberta-v3-small`. Its tokenizer was loaded using Hugging Face `AutoTokenizer`. DeBERTa-v3 uses a SentencePiece-based subword vocabulary.

The question and option were passed as a sentence pair:

```python
tokenizer(
    question,
    option,
    truncation=True,
    padding="max_length",
    max_length=320
)
```

The maximum length was set to **320 tokens**. Inputs longer than this limit were truncated, while shorter sequences were padded to the same length. Fixed-length batches make GPU processing straightforward and consistent.

### 4.2 Why This Tokenization Was Suitable

Subword tokenization is useful because it can represent rare, technical, and partially unseen words without requiring every full word to exist in the vocabulary. This was important for the competition because the MCQs covered many different subjects and contained technical terminology.

Pairwise tokenization also allowed the transformer to jointly read the question and one candidate answer. Self-attention could therefore model direct relationships between words in the question and words in the candidate option.

### 4.3 Other Text Representations Tested

Several alternatives were tested before selecting DeBERTa:

| Method | Representation |
|---|---|
| TF-IDF | Sparse word-frequency features |
| MiniLM | 384-dimensional dense sentence embeddings |
| BGE | 768-dimensional dense semantic embeddings |
| RoBERTa | Byte-level BPE transformer tokens |
| DeBERTa-v3 | SentencePiece-based transformer tokens |

TF-IDF was fast and surprisingly strong when combined with an MLP, but it did not model word order or context well. MiniLM and BGE captured semantic information, but fixed embeddings were less effective than fully fine-tuning a transformer for the question-option ranking task.

---

## 5. Modeling and Experimentation

### 5.1 Model 1 - TF-IDF + Cosine Similarity Baseline

The first model was a simple NLP baseline. A TF-IDF vectorizer converted the question prompt and answer options into sparse numerical vectors. Cosine similarity was then calculated between the question and each answer option.

The five options were ranked by cosine similarity, and the three highest-scoring options formed the final prediction.

This model was useful because it provided a low-cost reference point. However, TF-IDF mainly depends on word overlap. It cannot properly understand context, synonyms, or complex reasoning.

#### Validation Results

- Accuracy : **10.75%**
- Precision : **0.0986**
- Recall : **0.1140**
- F1 Score : **0.0973**
- Local MAP@3: **0.2783**
- Kaggle score: **0.30922**

### 5.2 Model 2 - MiniLM Sentence Transformer

The second model used `sentence-transformers/all-MiniLM-L6-v2`. MiniLM converts text into dense semantic embeddings. The prompt and each answer option were embedded independently and compared using cosine similarity.

Unlike TF-IDF, MiniLM can recognize semantically related text even when the exact words are different. This produced an improvement over the first baseline.

#### Validation Results

- Accuracy : **24.00%**
- Precision : **0.2413**
- Recall : **0.2484**
- F1 Score : **0.2398**
- Local MAP@3: **0.3771**
- Kaggle score: **0.38653**

The model still had an important limitation: the embeddings were fixed and the model was not trained specifically for the competition.

### 5.3 Model 3 - Custom TF-IDF + MLP

The third model was the first neural network built from scratch using PyTorch. Every question-option pair was converted to TF-IDF features with a maximum of 5,000 features. But after processing only 2761 features are formed.

#### Architecture

```text
Question + Answer Option
          |
          v
 TF-IDF Vector (2761)
          |
          v
 Linear: 2761 -> 512
          |
        ReLU
          |
     Dropout 0.30
          |
          v
 Linear: 512 -> 128
          |
        ReLU
          |
     Dropout 0.30
          |
          v
  Linear: 128 -> 2
          |
          v
Incorrect / Correct probability
```

The two output classes represented:

- `0` = incorrect option
- `1` = correct option

The model used CrossEntropyLoss, Adam optimization with a learning rate of `0.001`, batch size 64, and 10 training epochs.

The model produced a large improvement over cosine-similarity approaches.

#### Validation Results

- Accuracy : **92.75%**
- Precision : **0.9329**
- Recall : **0.9344**
- F1 Score : **0.9305**
- Local MAP@3: **0.9604**
- Kaggle score: **0.72901**

This experiment showed that supervised learning on question-option pairs was much more effective than simply measuring semantic similarity.

### 5.4 Model 3.1 - Tuned TF-IDF + MLP

The MLP was further tuned while keeping the same basic architecture. The main changes were:

- AdamW optimizer instead of Adam,
- learning rate reduced to `5e-4`,
- weight decay of `1e-4`,
- ReduceLROnPlateau learning-rate scheduler,
- training increased to a maximum of 20 epochs,
- best model selected using validation loss.

#### Validation Results

- Accuracy : **93.75%**
- Precision : **0.9433**
- Recall : **0.9426**
- F1 Score : **0.9397**
- Local MAP@3: **0.9654**
- Kaggle score: **0.73025**

This small improvement showed that regularization and training strategy can matter even when the architecture remains unchanged.

### 5.5 Model 4 - BGE Embeddings + MLP

The fourth experiment replaced TF-IDF with the pretrained `BAAI/bge-base-en-v1.5` sentence embedding model. Each question-option pair was converted into a **768-dimensional dense vector**.

The BGE model itself was used as a fixed feature extractor. The embeddings were then passed into an MLP.

#### Architecture

```text
Question + Answer Option
          |
          v
 BGE Embedding (768)
          |
          v
 Linear: 768 -> 512
          |
        ReLU
          |
     Dropout 0.30
          |
          v
 Linear: 512 -> 128
          |
        ReLU
          |
     Dropout 0.30
          |
          v
  Linear: 128 -> 2
          |
          v
Incorrect / Correct probability
```

Because one question produces one positive and four negative pairs, the training data had a 1:4 class imbalance. Weighted CrossEntropyLoss with weights `[1.0, 4.0]` was therefore used. The model was trained for a maximum of 20 epochs with AdamW (`5e-4`), weight decay (`1e-4`), batch size 64, and a ReduceLROnPlateau scheduler.

**Validation Results:**

- Accuracy : **87.25%**
- Precision : **0.8766**
- Recall : **0.8688**
- F1 Score : **0.8720**
- Local MAP@3: **0.91958**
- Kaggle score: **0.70074**

Although BGE is a stronger semantic representation than TF-IDF, this model performed below the TF-IDF MLP on the public leaderboard. This suggested that stronger embeddings alone do not guarantee better ranking performance.

### 5.6 Model 5 - Fine-Tuned RoBERTa-base

RoBERTa was the first fully fine-tuned transformer used in the project. RoBERTa is an encoder-only transformer based on BERT. It improves BERT pretraining through changes such as longer training, more data, dynamic masking, and removal of the next-sentence prediction objective.

For this task, RoBERTa was converted into a binary sequence classifier with two labels. Each question and option were tokenized as a pair. The probability of class `1` was used as the answer score.

#### Fine-Tuning Setup

| Setting | Value |
|---|---|
| Base model | `roberta-base` |
| Task | Pairwise binary classification |
| Maximum sequence length | 320 |
| Training batch size | 8 |
| Maximum epochs | 5 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Evaluation | Every epoch |
| Early stopping patience | 2 |
| Best checkpoint criterion | Validation F1 |
| Frozen layers | None; full model fine-tuned |
| Quantization | Not used |

The model was trained using the Hugging Face `Trainer` API. W&B recorded the training loss, validation loss, validation accuracy, and final metrics.

#### Validation results

- Accuracy : **97.50%**
- Precision : **0.9711**
- Recall : **0.9784**
- F1 Score : **0.9743**
- Local MAP@3 : **0.9825**
- Kaggle score : **0.73233**

RoBERTa produced excellent local validation results, but its leaderboard gain over the tuned MLP was small. This indicated a large difference between local validation and the hidden Kaggle test distribution.

### 5.7 Model 6 - Fine-Tuned DeBERTa-v3-small

The final and best model was `microsoft/deberta-v3-small`. DeBERTa improves transformer attention by representing content and position separately through **disentangled attention**. DeBERTa-v3 further improves pretraining by using replaced-token detection and gradient-disentangled embedding sharing.

This architecture was suitable for the MCQ task because the model needed to identify detailed relationships between the question and each possible answer.

#### Pairwise Classification Flow

```text
Question -------------------+
                            |
Answer Option --------------+--> DeBERTa-v3-small
                                      |
                                      v
                              Classification Head
                                      |
                                      v
                           P(incorrect), P(correct)
                                      |
                                      v
                        Rank five options by P(correct)
                                      |
                                      v
                             Return top 3 options
```

#### Fine-Tuning Setup

| Setting | Value |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Task | Pairwise binary classification |
| Maximum sequence length | 320 |
| Training batch size | 8 |
| Test batch size | 32 |
| Maximum epochs | 5 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Evaluation strategy | Every epoch |
| Saving strategy | Every epoch |
| Early stopping patience | 2 |
| Best checkpoint criterion | Validation F1 |
| Mixed precision | FP16 when CUDA was available |
| Frozen layers | None; full model fine-tuned |
| Quantization | Not used |

#### Validation Results

- Accuracy : **99.00%** 
- Precision : **0.9889**
- Recall : **0.9904** 
- F1 Score : **0.9893** 
- Local MAP@3 : **0.9950** 
- Kaggle score : **0.75602**

DeBERTa-v3-small was therefore selected as the final model.

---

## 6. Performance and Comparative Analysis

### 6.1 Evaluation Metrics

The main competition metric was **Mean Average Precision at 3 (MAP@3)**.

Because every question has only one correct answer, the score for one question is:

```text
Correct answer ranked 1st -> 1.000
Correct answer ranked 2nd -> 0.500
Correct answer ranked 3rd -> 0.333
Correct answer not in top 3 -> 0.000
```

The final MAP@3 is the average of these scores across all questions.

Additional metrics used during model development were:

- **Accuracy:** fraction of top-1 predictions that were correct.
- **Precision:** proportion of predicted positives that were correct.
- **Recall:** proportion of actual positives identified correctly.
- **F1 score:** harmonic mean of precision and recall.
- **Training/validation loss:** used to observe optimization and overfitting.

### 6.2 Comparative Results

| Model | Main Representation | Accuracy | Precision | Recall | F1 Score | Local MAP@3 | Kaggle Score |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Cosine | Sparse TF-IDF | 0.1075 | 0.0986 | 0.1140  |  0.0973 | 0.2783 | 0.30922 |
| MiniLM + Cosine | 384-d dense embedding | 0.2400 | 0.2413 | 0.2484 | 0.2398 | 0.3771 | 0.38653 |
| TF-IDF + MLP | 5,000 TF-IDF features | 0.9275 | 0.9329 |0.9344 | 0.9305 | 0.9604 | 0.72901 |
| Tuned TF-IDF + MLP | 5,000 TF-IDF features | 0.9375 | 0.9433 | 0.9426 | 0.9397 | 0.9654 | 0.73025 |
| BGE + MLP | 768-d BGE embedding | 0.8610| 0.5974 | 0.9350 | 0.7290 | 0.92417 | 0.70074 |
| RoBERTa-base | Fine-tuned transformer | 0.9750 | 0.9711 | 0.9784 | 0.9743 | 0.9825 | 0.73233 |
| **DeBERTa-v3-small** | **Fine-tuned transformer** | **0.9900** | **0.9889** | **0.9904** | **0.9893** | **0.9950** | **0.75602** |


### 6.3 Main Findings

The experiments showed a clear progression.

First, TF-IDF with cosine similarity performed poorly because lexical overlap is not enough for difficult MCQs. MiniLM improved the score by introducing semantic embeddings, but it still used a fixed similarity function rather than learning directly from the competition labels.

The largest early improvement came from the TF-IDF + MLP model. This showed that **supervised pairwise classification** was more important than simply using a sophisticated embedding model. The MLP learned patterns that cosine similarity could not learn.

BGE + MLP did not improve the Kaggle score. Although the BGE embeddings were semantically richer, the fixed embedding representation may have removed useful token-level interactions needed for fine answer discrimination.

Fine-tuning RoBERTa and DeBERTa allowed the entire language model to adapt to the task. DeBERTa gave the best result because its contextual attention mechanism could directly compare the question and option at token level.

### 6.4 Weights & Biases Tracking

W&B was used during model development to record training loss, validation loss, training/validation accuracy, and final evaluation metrics.

> ![Training Loss](image-7.png)           ![Validation loss](image-10.png) 

> ![Training Accuracy](image-8.png)         ![Validation Accuracy](image-9.png) 

> ![Training Loss of RoBErTa & DeBErTa Models](image-4.png)        ![Validation Loss of RoBErTa & DeBErTa Models ](image-5.png)

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

This project explored several approaches to solving ranked multiple-choice questions. The experiments started with simple word-overlap methods and gradually moved toward learned neural models and fine-tuned transformers.

The most important improvement came from reformulating the task as pairwise classification. Instead of trying to directly predict one of five labels, the models learned whether each individual question-option pair was correct or incorrect. The five probabilities could then be ranked naturally for MAP@3.

The custom TF-IDF MLP provided a strong result with relatively low computational cost. However, the best overall model was DeBERTa-v3-small, which achieved a validation MAP@3 of 0.9950 and the best recorded Kaggle score of 0.75602.

### 7.2 Key Learnings

The main learning from the project was that model architecture alone is not enough. The way the task is represented can strongly affect performance. Pairwise classification provided a much better learning signal than simple cosine similarity.

The project also provided practical experience with:

- text preprocessing and exploratory data analysis,
- TF-IDF and sentence embeddings,
- custom neural networks in PyTorch,
- GPU-based model training,
- Hugging Face tokenizers and Trainer,
- fine-tuning pretrained transformers,
- early stopping and checkpoint selection,
- W&B experiment tracking,
- ranking predictions for MAP@3,
- creating valid Kaggle submission files.

### 7.3 Challenges Faced

One challenge was the difference between local validation and Kaggle leaderboard performance. Some models looked extremely strong locally but produced smaller gains on the hidden test set.

Another challenge was class imbalance in the pairwise dataset. Every question produced one positive pair and four negative pairs. Weighted loss was tested in the BGE + MLP model to address this imbalance.

Computational cost also increased substantially when moving from TF-IDF and fixed embeddings to full transformer fine-tuning. Batch size, maximum sequence length, checkpoint saving, and GPU memory therefore had to be managed carefully.

The experimental RAG pipeline faced problems with knowledge-base quality and retrieval relevance.Some retrieved documents were duplicated or did not closely match the intended topic. As a result, the language model sometimes received irrelevant context. This reduced the reliability of the RAG approach.Instead of forcing an unsuccessful method into the final model, I decided to keep DeBERTa as the final solution and treat RAG as an important future research direction.

### 7.4 Future Work

Several improvements could be tested in future work:

1. Improve the RAG system.Build a cleaner knowledge base with reliable, topic-matched documents.
2. Use stratified or repeated cross-validation to obtain a more reliable validation estimate.
3. Train with hard negative options instead of treating every wrong option equally.
4. Test listwise ranking losses that directly compare all five options together.
5. Ensemble DeBERTa with the strong TF-IDF MLP because the models may make different errors.
6. Experiment with DeBERTa-v3-base if more GPU memory and training time are available.
7. Perform systematic hyperparameter tuning for learning rate, sequence length, and batch size.
8. Analyse errors by subject or question type to identify where the final model fails.

---

## 8. References

1. Hugging Face. **microsoft/deberta-v3-small Model Card.** https://huggingface.co/microsoft/deberta-v3-small

2. Hugging Face / Sentence Transformers. **all-MiniLM-L6-v2 Model Card.** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

3. Beijing Academy of Artificial Intelligence (BAAI). **bge-base-en-v1.5 Model Card.** https://huggingface.co/BAAI/bge-base-en-v1.5

4. Hugging Face. **Transformers Library.** https://huggingface.co/docs/transformers/

5. PyTorch. **PyTorch Documentation.** https://pytorch.org/docs/stable/index.html

6. Scikit-learn. **Machine Learning in Python.** https://scikit-learn.org/

7. Weights & Biases. **Experiment Tracking Documentation.** https://docs.wandb.ai/

8. Kaggle. **Smart MCQ Solver Challenge.** Competition dataset and evaluation description used in this project.

---
