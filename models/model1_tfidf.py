"""
Model 1: TF-IDF + Cosine Similarity
Smart MCQ Solver Challenge

"""

import re
import string
import pandas as pd
import wandb

from kaggle_secrets import UserSecretsClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===========================
# Configuration
# ===========================
TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
TEST_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv"

# ===========================
# W&B Login
# ===========================
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=api_key)

wandb.init(
    entity="25ds1000066-dl-genai-project",
    project="25ds1000066-t22026",
    name="Model1_TFIDF",
    config={
        "model":"TF-IDF",
        "similarity":"Cosine Similarity"
    }
)

# ===========================
# Load Data
# ===========================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# ===========================
# Text Cleaning
# ===========================
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_clean = train.copy()
test_clean = test.copy()

train_clean["prompt"] = train_clean["prompt"].apply(clean_text)
test_clean["prompt"] = test_clean["prompt"].apply(clean_text)

for option in ["A","B","C","D","E"]:
    train_clean[option] = train_clean[option].apply(clean_text)
    test_clean[option] = test_clean[option].apply(clean_text)

# ===========================
# Train / Validation Split
# ===========================
train_split, valid_split = train_test_split(
    train_clean,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

combined_text = (
    train_split["prompt"] + " " +
    train_split["A"] + " " +
    train_split["B"] + " " +
    train_split["C"] + " " +
    train_split["D"] + " " +
    train_split["E"]
)

vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(combined_text)

# ===========================
# MAP@3 Functions
# ===========================
def apk(actual, predicted, k=3):
    predicted = predicted[:k]
    for i,p in enumerate(predicted):
        if p == actual:
            return 1/(i+1)
    return 0

def mapk(actuals, predictions, k=3):
    return sum(apk(a,p,k) for a,p in zip(actuals,predictions))/len(actuals)

# ===========================
# Validation
# ===========================
top3_predictions = []
top1_predictions = []

for _, row in valid_split.iterrows():

    prompt_vec = vectorizer.transform([row["prompt"]])

    scores = []

    for option in ["A","B","C","D","E"]:

        option_vec = vectorizer.transform([row[option]])
        sim = cosine_similarity(prompt_vec, option_vec)[0][0]
        scores.append((option, sim))

    scores.sort(key=lambda x:x[1], reverse=True)

    top3_predictions.append([x[0] for x in scores[:3]])
    top1_predictions.append(scores[0][0])

actual_answers = valid_split["answer"]

tfidf_accuracy = accuracy_score(actual_answers, top1_predictions)
tfidf_precision = precision_score(actual_answers, top1_predictions, average="macro")
tfidf_recall = recall_score(actual_answers, top1_predictions, average="macro")
tfidf_f1 = f1_score(actual_answers, top1_predictions, average="macro")
tfidf_map3 = mapk(actual_answers, top3_predictions)

print("Validation Results")
print("-------------------")
print(f"Accuracy : {tfidf_accuracy:.4f}")
print(f"Precision: {tfidf_precision:.4f}")
print(f"Recall   : {tfidf_recall:.4f}")
print(f"F1 Score : {tfidf_f1:.4f}")
print(f"MAP@3    : {tfidf_map3:.4f}")

wandb.log({
    "Accuracy": tfidf_accuracy,
    "Precision": tfidf_precision,
    "Recall": tfidf_recall,
    "F1 Score": tfidf_f1,
    "MAP@3": tfidf_map3,
    "Kaggle Score": 0.30922
})

# ===========================
# Test Prediction
# ===========================
test_predictions = []

for _, row in test_clean.iterrows():

    prompt_vec = vectorizer.transform([row["prompt"]])

    scores = []

    for option in ["A","B","C","D","E"]:
        option_vec = vectorizer.transform([row[option]])
        sim = cosine_similarity(prompt_vec, option_vec)[0][0]
        scores.append((option, sim))

    scores.sort(key=lambda x:x[1], reverse=True)

    test_predictions.append([x[0] for x in scores[:3]])

submission = pd.DataFrame({
    "id": test["id"],
    "prediction": [" ".join(x) for x in test_predictions]
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission saved as submission.csv")

wandb.finish()
