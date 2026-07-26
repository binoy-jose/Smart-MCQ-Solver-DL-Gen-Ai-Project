"""
Model 3: MLP with TF-IDF Features
Smart MCQ Solver Challenge

This script is a standalone version of the Model 3 notebook.
"""

# ==========================
# Imports
# ==========================
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from kaggle_secrets import UserSecretsClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

# ==========================
# Configuration
# ==========================
TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
TEST_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv"
SAMPLE_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv"

# ==========================
# W&B Login
# ==========================
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("WANDB_API_KEY")
wandb.login(key=api_key)

wandb.init(
    entity="25ds1000066-dl-genai-project",
    project="25ds1000066-t22026",
    name="Model3_MLP_TFIDF",
    config={
        "model":"MLP",
        "input_features":"TF-IDF",
        "optimizer":"Adam",
        "loss_function":"CrossEntropyLoss"
    }
)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# ==========================
# Cleaning
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

train = train.copy()
test = test.copy()

train["prompt"] = train["prompt"].apply(clean_text)
test["prompt"] = test["prompt"].apply(clean_text)

for col in ["A","B","C","D","E"]:
    train[col] = train[col].apply(clean_text)
    test[col] = test[col].apply(clean_text)

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
    random_state=42,
    shuffle=True
)

# ==========================
# Pairwise Dataset
# ==========================
def create_pairwise_dataset(df):
    rows=[]
    for _,r in df.iterrows():
        ans=r["answer"]
        for op in ["A","B","C","D","E"]:
            rows.append({
                "text":r["prompt"]+" "+r[op],
                "label":1 if op==ans else 0
            })
    return pd.DataFrame(rows)

pairwise_train=create_pairwise_dataset(train_questions)
pairwise_valid=create_pairwise_dataset(valid_questions)

# ==========================
# TF-IDF
# ==========================
vectorizer=TfidfVectorizer(stop_words="english",max_features=5000)

X_train=vectorizer.fit_transform(pairwise_train["text"]).toarray()
X_valid=vectorizer.transform(pairwise_valid["text"]).toarray()

y_train=pairwise_train["label"].values
y_valid=pairwise_valid["label"].values

X_train=torch.FloatTensor(X_train)
X_valid=torch.FloatTensor(X_valid)
y_train=torch.LongTensor(y_train)
y_valid=torch.LongTensor(y_valid)

class MCQDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

train_loader=DataLoader(MCQDataset(X_train,y_train),batch_size=64,shuffle=True)
valid_loader=DataLoader(MCQDataset(X_valid,y_valid),batch_size=64,shuffle=False)

class MLPClassifier(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,2)
        )
    def forward(self,x):
        return self.network(x)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MLPClassifier(X_train.shape[1]).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

# ==========================
# Training
# ==========================
for epoch in range(10):
    model.train()
    train_loss=0
    train_correct=0
    total=0

    for x,y in train_loader:
        x=x.to(device)
        y=y.to(device)

        optimizer.zero_grad()
        out=model(x)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        pred=out.argmax(1)
        total+=y.size(0)
        train_correct+=(pred==y).sum().item()

    model.eval()
    valid_loss=0
    valid_correct=0
    valid_total=0

    with torch.no_grad():
        for x,y in valid_loader:
            x=x.to(device)
            y=y.to(device)
            out=model(x)
            loss=criterion(out,y)
            valid_loss+=loss.item()
            pred=out.argmax(1)
            valid_total+=y.size(0)
            valid_correct+=(pred==y).sum().item()

    wandb.log({
        "Epoch":epoch+1,
        "Training Loss":train_loss/len(train_loader),
        "Training Accuracy":train_correct/total,
        "Validation Loss":valid_loss/len(valid_loader),
        "Validation Accuracy":valid_correct/valid_total
    })

# ==========================
# Validation Prediction
# ==========================
model.eval()
top3=[]
top1=[]

with torch.no_grad():
    for _,row in valid_questions.iterrows():
        scores=[]
        for op in ["A","B","C","D","E"]:
            txt=row["prompt"]+" "+row[op]
            vec=torch.FloatTensor(vectorizer.transform([txt]).toarray()).to(device)
            prob=torch.softmax(model(vec),dim=1)[0][1].item()
            scores.append((op,prob))
        scores.sort(key=lambda x:x[1],reverse=True)
        top3.append([x[0] for x in scores[:3]])
        top1.append(scores[0][0])

acc=accuracy_score(valid_questions["answer"],top1)
prec=precision_score(valid_questions["answer"],top1,average="macro")
rec=recall_score(valid_questions["answer"],top1,average="macro")
f1=f1_score(valid_questions["answer"],top1,average="macro")
m3=mapk(valid_questions["answer"],top3)

wandb.log({
    "Accuracy":acc,
    "Precision":prec,
    "Recall":rec,
    "F1 Score":f1,
    "MAP@3":m3,
    "Kaggle Score":0.72901
})

# ==========================
# Test Prediction
# ==========================
preds=[]
with torch.no_grad():
    for _,row in test.iterrows():
        scores=[]
        for op in ["A","B","C","D","E"]:
            txt=row["prompt"]+" "+row[op]
            vec=torch.FloatTensor(vectorizer.transform([txt]).toarray()).to(device)
            prob=torch.softmax(model(vec),dim=1)[0][1].item()
            scores.append((op,prob))
        scores.sort(key=lambda x:x[1],reverse=True)
        preds.append(" ".join([x[0] for x in scores[:3]]))

sample["Prediction"]=preds
sample.to_csv("submission.csv",index=False)

wandb.finish()
print("Done.")
