import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import Process, Queue

# Argument parsing for device
parser = argparse.ArgumentParser(description='Titanic PyTorch Training')
parser.add_argument('--device', type=str, choices=['cpu','cuda'], default=None,
                    help='Compute device to use (cpu or cuda)')
args = parser.parse_args()

# Determine device
def get_device():
    if args.device:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise SystemError('CUDA specified but no GPU available')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

device = get_device()
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

# Feature engineering
def preprocess(df):
    print(f"[Preprocess] Input shape: {df.shape}")
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    title_map = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}
    df['Title'] = df['Title'].map(title_map).fillna(4).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Deck'] = df['Cabin'].fillna('U').str[0]
    deck_map = {d:i for i,d in enumerate(list('ABCDEFGU'))}
    df['Deck'] = df['Deck'].map(deck_map).fillna(deck_map['U']).astype(int)
    df['TicketPrefix'] = (
        df['Ticket']
        .str.replace(r'\d','',regex=True)
        .str.replace(r'\.', '', regex=True)
        .str.replace('/', '', regex=True)
        .str.strip()
        .str.split()
        .str[0]
    )
    df['TicketPrefix'] = df['TicketPrefix'].replace('', 'NONE')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['AgeBin'] = pd.qcut(df['Age'], 4, labels=False)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    print(f"[Preprocess] Output shape: {df.shape}")
    return df

# Prepare global preprocessor and input dimension
def build_preprocessor(train_df, features):
    # Use sparse_output=False for OneHotEncoder in recent sklearn
    pre = ColumnTransformer([
        ('num', StandardScaler(), ['FamilySize']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
    ])
    pre.fit(train_df[features])
    input_dim = pre.transform(train_df[features]).shape[1]
    print(f"[Preprocessor] Number of input features: {input_dim}")
    return pre, input_dim

# Define model builder
def make_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
        nn.Linear(64, 1), nn.Sigmoid()
    )

# Training/Evaluation routines
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            preds = model(xb)
            total_loss += criterion(preds, yb).item() * xb.size(0)
            correct += ((preds > 0.5) == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Run experiment for a given epoch budget
def run_experiment(max_epochs, pre, input_dim, queue=None):
    print(f"[Experiment {max_epochs}] Running on {device}")
    train_df = preprocess(pd.read_csv('train.csv'))
    features = ['Pclass','Sex','AgeBin','FareBin','FamilySize','Title','Deck','TicketPrefix','Embarked']
    X = pre.transform(train_df[features])
    y = train_df['Survived'].values
    X = X.toarray() if hasattr(X, 'toarray') else X
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    loader_tr = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
    loader_vl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=32)

    model = make_model(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_acc, counter = 0.0, 0
    patience = 2000
    for epoch in range(1, max_epochs+1):
        loss_tr = train_epoch(model, loader_tr, criterion, optimizer, device)
        loss_vl, acc_vl = eval_epoch(model, loader_vl, criterion, device)
        if acc_vl > best_acc:
            print(f"New best val_acc: {acc_vl:.4f} at epoch {epoch} in exp {max_epochs}")
            best_acc, counter = acc_vl, 0
            torch.save(model.state_dict(), f'best_model_{max_epochs}.pth')
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping exp {max_epochs} at epoch {epoch}")
            break
    if queue: queue.put((max_epochs, best_acc))

# Main
if __name__ == '__main__':
    print("Building preprocessor and launching experiments...")
    df_full = preprocess(pd.read_csv('train.csv'))
    features = ['Pclass','Sex','AgeBin','FareBin','FamilySize','Title','Deck','TicketPrefix','Embarked']
    pre, input_dim = build_preprocessor(df_full, features)

    epochs_list = [300, 600, 1000]
    q = Queue(); procs = []
    for e in epochs_list:
        p = Process(target=run_experiment, args=(e, pre, input_dim, q))
        p.start(); procs.append(p)
    results = [q.get() for _ in epochs_list]
    for p in procs: p.join()

    best_epochs, best_acc = max(results, key=lambda x: x[1])
    print(f"Best config: {best_epochs} epochs with val_acc={best_acc:.4f}")

    # Final prediction
    model = make_model(input_dim).to(device)
    model.load_state_dict(torch.load(f'best_model_{best_epochs}.pth', map_location=device))
    model.eval()
    test_df = preprocess(pd.read_csv('test.csv'))
    X_test = pre.transform(test_df[features])
    X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad(): preds = (model(X_test_tensor)>0.5).int().cpu().numpy().reshape(-1)

    out_dir = os.path.join('submissions','submission_pt'); os.makedirs(out_dir, exist_ok=True)
    submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':preds})
    fname = 'submission_pt.csv'; i=2
    while os.path.exists(os.path.join(out_dir,fname)): fname=f'submission_pt_{i}.csv'; i+=1
    submission_path=os.path.join(out_dir,fname)
    submission.to_csv(submission_path,index=False)
    print(f"Saved submission: {submission_path}")

    # cleanup
    for e in epochs_list:
        ckpt=f'best_model_{e}.pth'
        if os.path.exists(ckpt): os.remove(ckpt); print(f"Removed checkpoint {ckpt}")
