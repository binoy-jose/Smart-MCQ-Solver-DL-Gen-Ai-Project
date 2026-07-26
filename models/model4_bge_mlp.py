"""
Model 4: MLP with BGE Embeddings
Smart MCQ Solver Challenge

This script is a standalone version of the Model 4 notebook.
"""

# ==========================
# Imports
# ==========================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from kaggle_secrets import UserSecretsClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

# ==========================
# Configuration
# ==========================
TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
TEST_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv"
SAMPLE_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# ==========================
# W&B Login
# ==========================
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("WANDB_API_KEY")
wandb.login(key=api_key)

wandb.init(
    entity="25ds1000066-dl-genai-project",
    project="25ds1000066-t22026",
    name="Model4_BGE_MLP_v1",
    config={
        "model":"MLP",
        "input_features":"BGE Embeddings",
        "embedding_model":EMBEDDING_MODEL,
        "embedding_dimension":768,
        "architecture":"768-512-128-2",
        "dropout":0.3,
        "batch_size":BATCH_SIZE,
        "epochs":EPOCHS,
        "learning_rate":LEARNING_RATE,
        "optimizer":"AdamW",
        "weight_decay":WEIGHT_DECAY,
        "loss_function":"Weighted CrossEntropyLoss",
        "scheduler":"ReduceLROnPlateau",
        "normalize_embeddings":True
    }
)

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
    random_state=42,
    stratify=train["answer"]
)

# ==========================
# Pairwise Dataset
# ==========================
def create_embedding_dataset(df,is_train=True):
    texts=[]
    labels=[]

    for _,row in df.iterrows():
        question=row["prompt"]
        for op in ["A","B","C","D","E"]:
            texts.append(f"Question: {question} [SEP] Option: {row[op]}")
            if is_train:
                labels.append(1 if op==row["answer"] else 0)

    if is_train:
        return texts,labels
    return texts

embedding_train,y_train=create_embedding_dataset(train_questions)
embedding_valid,y_valid=create_embedding_dataset(valid_questions)

# ==========================
# BGE Embeddings
# ==========================
bge_model=SentenceTransformer(EMBEDDING_MODEL)

print("Generating training embeddings...")
X_train=bge_model.encode(
    embedding_train,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Generating validation embeddings...")
X_valid=bge_model.encode(
    embedding_valid,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save("bge_train_embeddings.npy",X_train)
np.save("bge_valid_embeddings.npy",X_valid)

X_train=torch.FloatTensor(X_train)
X_valid=torch.FloatTensor(X_valid)
y_train=torch.LongTensor(y_train)
y_valid=torch.LongTensor(y_valid)

class EmbeddingDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

train_loader=DataLoader(
    EmbeddingDataset(X_train,y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
valid_loader=DataLoader(
    EmbeddingDataset(X_valid,y_valid),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class BGEMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(768,512),
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
model=BGEMLP().to(device)

class_weights=torch.tensor([1.0,4.0],dtype=torch.float32).to(device)
criterion=nn.CrossEntropyLoss(weight=class_weights)
optimizer=optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ==========================
# Training
# ==========================
best_valid_loss=float("inf")

for epoch in range(EPOCHS):
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

    avg_train_loss=train_loss/len(train_loader)
    avg_valid_loss=valid_loss/len(valid_loader)
    train_accuracy=train_correct/total
    validation_accuracy=valid_correct/valid_total

    if avg_valid_loss<best_valid_loss:
        best_valid_loss=avg_valid_loss
        torch.save(model.state_dict(),"best_bge_mlp_model.pth")

    scheduler.step(avg_valid_loss)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Training Loss: {avg_train_loss:.4f} | "
        f"Training Acc: {train_accuracy:.4f} | "
        f"Valid Loss: {avg_valid_loss:.4f} | "
        f"Valid Acc: {validation_accuracy:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    wandb.log({
        "Epoch":epoch+1,
        "Training Loss":avg_train_loss,
        "Training Accuracy":train_accuracy,
        "Validation Loss":avg_valid_loss,
        "Validation Accuracy":validation_accuracy
    })

# ==========================
# Validation Prediction
# ==========================
model.load_state_dict(
    torch.load("best_bge_mlp_model.pth",map_location=device)
)
model.eval()

validation_probabilities=[]
validation_predictions=[]
validation_labels=[]

with torch.no_grad():
    for x,y in valid_loader:
        x=x.to(device)
        out=model(x)
        probabilities=torch.softmax(out,dim=1)
        pred=out.argmax(1)

        validation_probabilities.extend(probabilities[:,1].cpu().numpy())
        validation_predictions.extend(pred.cpu().numpy())
        validation_labels.extend(y.numpy())

acc=accuracy_score(validation_labels,validation_predictions)
prec=precision_score(validation_labels,validation_predictions)
rec=recall_score(validation_labels,validation_predictions)
f1=f1_score(validation_labels,validation_predictions)

top3=[]
actuals=[]
probability_index=0

for _,row in valid_questions.iterrows():
    scores=[]
    for op in ["A","B","C","D","E"]:
        scores.append((op,validation_probabilities[probability_index]))
        probability_index+=1
    scores.sort(key=lambda x:x[1],reverse=True)
    top3.append([x[0] for x in scores[:3]])
    actuals.append(row["answer"])

m3=mapk(actuals,top3)

print(f"Accuracy : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"MAP@3 : {m3:.5f}")

wandb.log({
    "Accuracy":acc,
    "Precision":prec,
    "Recall":rec,
    "F1 Score":f1,
    "MAP@3":m3,
    "Kaggle Score":0.70074
})

# ==========================
# Save Checkpoint
# ==========================
torch.save({
    "epoch":EPOCHS-1,
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
    "scheduler_state_dict":scheduler.state_dict(),
    "best_valid_loss":best_valid_loss
},"best_checkpoint.pth")

# ==========================
# Test Prediction
# ==========================
embedding_test=create_embedding_dataset(test,is_train=False)

print("Generating test embeddings...")
X_test=bge_model.encode(
    embedding_test,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

X_test=torch.FloatTensor(X_test)
test_loader=DataLoader(X_test,batch_size=BATCH_SIZE,shuffle=False)

test_probabilities=[]

with torch.no_grad():
    for x in test_loader:
        x=x.to(device)
        out=model(x)
        probabilities=torch.softmax(out,dim=1)
        test_probabilities.extend(probabilities[:,1].cpu().tolist())

preds=[]
probability_index=0

for _,row in test.iterrows():
    scores=[]
    for op in ["A","B","C","D","E"]:
        scores.append((op,test_probabilities[probability_index]))
        probability_index+=1
    scores.sort(key=lambda x:x[1],reverse=True)
    preds.append(" ".join([x[0] for x in scores[:3]]))

sample["Prediction"]=preds
sample.to_csv("submission.csv",index=False)

wandb.finish()
print("Done.")
