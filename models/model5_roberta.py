"""
Model 5: Fine-tuned RoBERTa Pairwise Classifier
Smart MCQ Solver Challenge

This script is a standalone version of the Model 5 training and submission notebooks.
"""

# ==========================
# Imports
# ==========================
import json
import random
import numpy as np
import pandas as pd
import torch
import wandb

from kaggle_secrets import UserSecretsClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

# ==========================
# Configuration
# ==========================
TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
TEST_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv"
SAMPLE_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv"

MODEL_NAME = "roberta-base"
SAVE_PATH = "/kaggle/working/roberta_model"

MAX_LENGTH = 320
BATCH_SIZE = 8
TEST_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
SEED = 42

# ==========================
# Reproducibility
# ==========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==========================
# W&B Login
# ==========================
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("WANDB_API_KEY")
wandb.login(key=api_key)

wandb.init(
    entity="25ds1000066-dl-genai-project",
    project="25ds1000066-t22026",
    name="Model5_RoBERTa",
    config={
        "model":"RoBERTa-base",
        "architecture":"Pairwise Classification",
        "optimizer":"AdamW",
        "learning_rate":LEARNING_RATE,
        "epochs":EPOCHS,
        "batch_size":BATCH_SIZE,
        "max_length":MAX_LENGTH,
        "loss_function":"CrossEntropyLoss",
        "early_stopping_patience":2
    }
)

# ==========================
# Load Data
# ==========================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# ==========================
# MAP@3
# ==========================
def apk(actual,predicted,k=3):
    predicted = predicted[:k]
    for i,p in enumerate(predicted):
        if p==actual:
            return 1/(i+1)
    return 0

def mapk(actuals,preds,k=3):
    return sum(apk(a,p,k) for a,p in zip(actuals,preds))/len(actuals)

# ==========================
# Split
# ==========================
train_questions, valid_questions = train_test_split(
    train,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

print("Training Questions:", len(train_questions))
print("Validation Questions:", len(valid_questions))

# ==========================
# Pairwise Dataset
# ==========================
def create_pairwise_dataset(df,is_train=True):
    rows=[]

    for _,row in df.iterrows():
        question=str(row["prompt"])

        for op in ["A","B","C","D","E"]:
            pair={
                "question":question,
                "option":str(row[op]),
                "option_name":op
            }

            if is_train:
                pair["label"]=int(op==row["answer"])
                pair["correct_option"]=row["answer"]

            rows.append(pair)

    return pd.DataFrame(rows)

pairwise_train=create_pairwise_dataset(train_questions,is_train=True)
pairwise_valid=create_pairwise_dataset(valid_questions,is_train=True)

print("Training Samples:", len(pairwise_train))
print("Validation Samples:", len(pairwise_valid))

# ==========================
# Tokenizer
# ==========================
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        examples["option"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# ==========================
# Hugging Face Datasets
# ==========================
train_dataset=Dataset.from_pandas(pairwise_train)
valid_dataset=Dataset.from_pandas(pairwise_valid)

train_dataset=train_dataset.map(
    tokenize_function,
    batched=True
)

valid_dataset=valid_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset=train_dataset.rename_column("label","labels")
valid_dataset=valid_dataset.rename_column("label","labels")

train_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)

valid_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)

# ==========================
# RoBERTa Model
# ==========================
model=AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(device)

# ==========================
# Trainer Metrics
# ==========================
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=1)

    return {
        "accuracy":accuracy_score(labels,predictions),
        "precision":precision_score(
            labels,predictions,average="binary",zero_division=0
        ),
        "recall":recall_score(
            labels,predictions,average="binary",zero_division=0
        ),
        "f1":f1_score(
            labels,predictions,average="binary",zero_division=0
        )
    }

class WandbMetricsCallback(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs is None:
            return

        metrics={}

        if "loss" in logs:
            metrics["Training Loss"]=logs["loss"]

        if "eval_loss" in logs:
            metrics["Validation Loss"]=logs["eval_loss"]

        if "eval_accuracy" in logs:
            metrics["Validation Accuracy"]=logs["eval_accuracy"]

        if metrics:
            wandb.log(metrics)

# ==========================
# Training Arguments
# ==========================
training_args=TrainingArguments(
    output_dir="./roberta_checkpoints",
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",
    run_name="Model5_RoBERTa",
    seed=SEED,
    remove_unused_columns=True,
    fp16=False,
    bf16=False
)

# ==========================
# Trainer
# ==========================
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        WandbMetricsCallback()
    ]
)

# ==========================
# Training
# ==========================
print("Starting RoBERTa Training...")

trainer.train()

print("Training Completed Successfully!")

# ==========================
# Validation Prediction
# ==========================
validation_output=trainer.predict(valid_dataset)
validation_logits=validation_output.predictions

validation_probabilities=torch.softmax(
    torch.tensor(validation_logits),
    dim=1
).numpy()

top1=[]
top3=[]

for i in range(0,len(pairwise_valid),5):
    scores=validation_probabilities[i:i+5,1]
    option_names=pairwise_valid.iloc[i:i+5]["option_name"].tolist()

    ranking=sorted(
        zip(option_names,scores),
        key=lambda x:x[1],
        reverse=True
    )

    top1.append(ranking[0][0])
    top3.append([x[0] for x in ranking[:3]])

actuals=valid_questions["answer"].tolist()

acc=accuracy_score(actuals,top1)
prec=precision_score(actuals,top1,average="macro",zero_division=0)
rec=recall_score(actuals,top1,average="macro",zero_division=0)
f1=f1_score(actuals,top1,average="macro",zero_division=0)
m3=mapk(actuals,top3)

print(f"Accuracy : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"MAP@3 : {m3:.4f}")

wandb.log({
    "Accuracy":acc,
    "Precision":prec,
    "Recall":rec,
    "F1 Score":f1,
    "MAP@3":m3,
    "Kaggle Score":0.73233
})

# ==========================
# Save Fine-tuned Model
# ==========================
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

metrics={
    "Accuracy":float(acc),
    "Precision":float(prec),
    "Recall":float(rec),
    "F1 Score":float(f1),
    "MAP@3":float(m3)
}

with open(f"{SAVE_PATH}/metrics.json","w") as file:
    json.dump(metrics,file,indent=4)

print("RoBERTa model and metrics saved successfully.")

# ==========================
# Reload Saved Model
# ==========================
del trainer
del model

if torch.cuda.is_available():
    torch.cuda.empty_cache()

tokenizer=AutoTokenizer.from_pretrained(SAVE_PATH)

model=AutoModelForSequenceClassification.from_pretrained(
    SAVE_PATH
)

model.to(device)
model.eval()

print("Saved RoBERTa model loaded successfully.")

# ==========================
# Test Prediction
# ==========================
pairwise_test=create_pairwise_dataset(
    test,
    is_train=False
)

test_dataset=Dataset.from_pandas(pairwise_test)

test_dataset=test_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask"]
)

test_loader=DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False
)

test_probabilities=[]

with torch.no_grad():
    for batch in test_loader:
        batch={
            key:value.to(device)
            for key,value in batch.items()
        }

        outputs=model(**batch)
        probabilities=torch.softmax(outputs.logits,dim=1)

        test_probabilities.extend(
            probabilities[:,1].cpu().numpy()
        )

# ==========================
# Top-3 Predictions
# ==========================
preds=[]

for i in range(0,len(pairwise_test),5):
    scores=test_probabilities[i:i+5]
    option_names=pairwise_test.iloc[i:i+5]["option_name"].tolist()

    ranking=sorted(
        zip(option_names,scores),
        key=lambda x:x[1],
        reverse=True
    )

    preds.append(" ".join([x[0] for x in ranking[:3]]))

print("First Predictions:", preds[:5])

# ==========================
# Submission
# ==========================
sample["Prediction"]=preds
sample.to_csv("submission.csv",index=False)

wandb.finish()

print("submission.csv created successfully.")
print("Done.")
