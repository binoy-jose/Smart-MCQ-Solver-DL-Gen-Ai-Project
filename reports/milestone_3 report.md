# Milestone 3 Report

## Objective

The objective of Milestone 3 was to understand the core concepts of Retrieval-Augmented Generation (RAG). The milestone focused on retrieving relevant information from a knowledge base using vector search, improving retrieval quality through Cross-Encoder reranking, and evaluating how external context influences the predictions of a large language model.

---

## Dataset

The same **Smart MCQ Solver** dataset was used, consisting of:

- Question prompt
- Five answer options (A, B, C, D, E)
- Correct answer label (training set)

A knowledge base was created by storing the correct answer text from every training sample. These documents were embedded and indexed using FAISS for efficient similarity search.

---

## Tasks Completed

- Created a knowledge base using the correct answers from the training dataset.
- Generated sentence embeddings using the **all-MiniLM-L6-v2** model.
- Built a **FAISS vector index** for efficient document retrieval.
- Performed semantic retrieval of relevant documents for a given question.
- Evaluated retrieval quality using **Hit Rate**.
- Applied **Cross-Encoder (ms-marco-MiniLM-L-6-v2)** for reranking retrieved documents.
- Compared retrieval rankings before and after reranking.
- Built a Retrieval-Augmented Generation (RAG) pipeline by injecting retrieved context into the prompt.
- Compared zero-shot classification with and without retrieved context.
- Demonstrated the effect of correct and incorrect retrieval (Adversarial RAG).
- Evaluated the complete RAG pipeline using **MAP@3**.

---

## Methods Used

- Python
- FAISS
- Sentence Transformers
- all-MiniLM-L6-v2
- Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- Hugging Face Transformers
- BART Large MNLI (Zero-Shot Classification)
- BERT Tokenizer
- PyTorch

---

## Retrieval Pipeline

The RAG pipeline implemented during this milestone consisted of the following stages:

1. Embed the question using **all-MiniLM-L6-v2**.
2. Retrieve the Top-K most similar documents using **FAISS**.
3. Rerank the retrieved documents using the **Cross-Encoder**.
4. Select the highest-ranked document as context.
5. Augment the prompt with the retrieved context.
6. Perform zero-shot answer prediction using **facebook/bart-large-mnli**.
7. Evaluate the predictions using **MAP@3**.

---

## Evaluation Metrics

The following evaluation metrics were explored:

- **Hit Rate** – Measures whether the correct supporting document appears among the retrieved documents.
- **MAP@3 (Mean Average Precision at 3)** – Measures how highly the correct answer is ranked among the top three predicted options.

---

## Model Performance

**Retrieval Model:** all-MiniLM-L6-v2 + FAISS

- Efficient semantic retrieval of relevant documents from the knowledge base.
- Evaluated using Hit Rate.

**Reranking Model:** Cross-Encoder (ms-marco-MiniLM-L-6-v2)

- Improved retrieval quality by reordering the retrieved documents based on semantic relevance.

**Prediction Model:** facebook/bart-large-mnli

- Compared predictions under three settings:
  - Zero-shot (no context)
  - RAG with correct retrieved context
  - RAG with incorrect (adversarial) context

**Final Pipeline**

FAISS Retrieval → Cross-Encoder Reranking → RAG Prompt → BART Zero-Shot Classification

Evaluated using **MAP@3** on the selected evaluation samples.

---

## Key Observations

- Semantic retrieval provided relevant supporting information for many questions.
- Cross-Encoder reranking improved the ordering of retrieved documents compared to FAISS alone.
- Injecting the correct context generally increased the confidence of the correct prediction.
- Providing incorrect context reduced prediction confidence and demonstrated the importance of retrieval quality.
- The quality of retrieved documents directly affected the overall performance of the RAG pipeline.

---

## Outcome

Milestone 3 introduced the complete Retrieval-Augmented Generation (RAG) workflow. By combining semantic retrieval, Cross-Encoder reranking, and zero-shot classification, the pipeline demonstrated how external knowledge can improve question answering performance. The milestone also highlighted the importance of accurate retrieval, showing that incorrect context can significantly degrade model predictions. These concepts provide the foundation for building more advanced retrieval-based and generative AI systems in future milestones.
