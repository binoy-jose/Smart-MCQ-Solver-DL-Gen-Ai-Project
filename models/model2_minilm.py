"""
Model 2: Sentence Transformer (MiniLM)
Smart MCQ Solver Challenge
"""

import pandas as pd
import wandb

from kaggle_secrets import UserSecretsClient
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# W&B Login
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=api_key)

wandb.init(
    entity="25ds1000066-dl-genai-project",
    project="25ds1000066-t22026",
    name="Model2_MiniLM",
    config={
        "model":"MiniLM",
        "embedding_model":"all-MiniLM-L6-v2",
        "similarity":"Cosine Similarity"
    }
)
# Cleaning Data
# ---------------
import string
import re

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_clean = train.copy()
test_clean = test.copy()

train_clean["prompt"] = train_clean["prompt"].apply(clean_text)
test_clean["prompt"] = test_clean["prompt"].apply(clean_text)

for option in ["A","B","C","D","E"]:
    train_clean[option] = train_clean[option].apply(clean_text)
    test_clean[option] = test_clean[option].apply(clean_text)

from sklearn.model_selection import train_test_split

# Spliting the data
# ----------------------
train_split, valid_split = train_test_split(
    train_clean,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(train_split.shape)
print(valid_split.shape)

# Load Model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- Validation ----------------

top3_predictions = []
top1_predictions = []

for _, row in valid_split.iterrows():

    prompt_embedding = model.encode(
        row["prompt"],
        convert_to_tensor=True
    )

    similarities = []

    for option in ["A","B","C","D","E"]:

        option_embedding = model.encode(
            row[option],
            convert_to_tensor=True
        )

        score = util.cos_sim(
            prompt_embedding,
            option_embedding
        ).item()

        similarities.append((option, score))

    similarities.sort(
        key=lambda x:x[1],
        reverse=True
    )

    top3_predictions.append(
        [x[0] for x in similarities[:3]]
    )

    top1_predictions.append(
        similarities[0][0]
    )

actual = valid_split["answer"]

accuracy = accuracy_score(actual, top1_predictions)
precision = precision_score(actual, top1_predictions, average="macro")
recall = recall_score(actual, top1_predictions, average="macro")
f1 = f1_score(actual, top1_predictions, average="macro")
map3 = mapk(actual, top3_predictions)

print("MiniLM Validation")
print("-----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"MAP@3    : {map3:.4f}")

wandb.log({
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "MAP@3": map3,
    "Kaggle Score":0.38653
})

# ---------------- Test Prediction ----------------

test_predictions = []

for _, row in test.iterrows():

    prompt_embedding = model.encode(
        row["prompt"],
        convert_to_tensor=True
    )

    similarities = []

    for option in ["A","B","C","D","E"]:

        option_embedding = model.encode(
            row[option],
            convert_to_tensor=True
        )

        score = util.cos_sim(
            prompt_embedding,
            option_embedding
        ).item()

        similarities.append((option, score))

    similarities.sort(
        key=lambda x:x[1],
        reverse=True
    )

    top3 = [x[0] for x in similarities[:3]]

    test_predictions.append(" ".join(top3))

# Works with either sample submission or test ids
try:
    submission = sample.copy()
    submission["Prediction"] = test_predictions
except NameError:
    submission = pd.DataFrame({
        "id": test["id"],
        "prediction": test_predictions
    })

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully.")

wandb.finish()
