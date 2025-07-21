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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import Process, Queue, Manager
from tqdm import trange

# Argument parsing for compute device
def parse_args():
    parser = argparse.ArgumentParser(description='Titanic PyTorch Training')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help='Compute device to use (cpu or cuda)')
    return parser.parse_args()

args = parse_args()

def get_device():
    if args.device:
        dev = torch.device(args.device)
        if dev.type == 'cuda' and not torch.cuda.is_available():
            raise SystemError('CUDA specified but not available')
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dev

# Select device
device = get_device()
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

# Training and evaluation loops
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb).squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb).squeeze()
            loss = criterion(out, yb)
            total_loss += loss.item() * Xb.size(0)
            preds = (out > 0.5).float()
            correct += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Data preprocessing
def preprocess(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}).fillna(4).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Deck'] = df['Cabin'].fillna('U').str[0].map({d:i for i,d in enumerate('ABCDEFGU')}).fillna(6).astype(int)
    df['TicketPrefix'] = (
        df['Ticket']
          .str.replace(r'\d+', '', regex=True)
          .str.replace(r'\.', '', regex=True)
          .str.replace('/', '', regex=True)
          .str.strip().str.split().str[0]
          .replace('', 'NONE')
    )
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['AgeBin'] = pd.qcut(df['Age'], 4, labels=False)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    df['Embarked'] = df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2}).astype(int)
    df['Pclass_Sex'] = df['Pclass'] * df['Sex']
    df['Age_Fare'] = df['Age'] * df['Fare']
    return df

# Preprocessor setup
def build_preprocessor(df, cat_features, num_features):
    ct = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])
    ct.fit(df)
    inp_dim = ct.transform(df).shape[1]
    print(f"[Preprocessor] input dim: {inp_dim}")
    return ct, inp_dim

# Model definition
def make_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
        nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
        nn.Linear(64, 1), nn.Sigmoid()
    )

# Single experiment runner
def run_experiment(epochs, ct, dim, cat_feats, num_feats, global_best, lock, q=None):
    print(f"[Start] Exp {epochs} on {device}")
    df_tr = preprocess(pd.read_csv('train.csv'))
    X = ct.transform(df_tr)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    y = df_tr['Survived'].values
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    lt = DataLoader(TensorDataset(
                  torch.tensor(X_t, dtype=torch.float32),
                  torch.tensor(y_t, dtype=torch.float32)),
                  batch_size=64, shuffle=True)
    lv = DataLoader(TensorDataset(
                  torch.tensor(X_v, dtype=torch.float32),
                  torch.tensor(y_v, dtype=torch.float32)),
                  batch_size=64)
    model = make_model(dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10000)
    crit = nn.BCELoss()
    best_acc_local, stagn = 0.0, 0
    for e in trange(1, epochs+1, desc=f"Exp {epochs}", leave=False):
        tr_loss = train_epoch(model, lt, crit, opt, device)
        vl_loss, vl_acc = eval_epoch(model, lv, crit, device)
        sched.step(vl_acc)
        if vl_acc > best_acc_local:
            best_acc_local, stagn = vl_acc, 0
            torch.save(model.state_dict(), 'best_model.pth')
            with lock:
                if vl_acc > global_best.value:
                    global_best.value = vl_acc
                    print(f"[Global New Best] Exp {epochs} epoch {e}: tr_loss={tr_loss:.4f}, vl_loss={vl_loss:.4f}, vl_acc={vl_acc:.4f}")
        else:
            stagn += 1
        if stagn >= 5000:
            print(f"[Exp {epochs}] Early stop at epoch {e} (no improvement for {stagn} iters)")
            break
    if q:
        q.put((epochs, best_acc_local))

# Main execution
if __name__ == '__main__':
    df0 = preprocess(pd.read_csv('train.csv'))
    cat_feats = ['Pclass','Sex','AgeBin','FareBin','FamilySize','Title','Deck',
                 'TicketPrefix','Embarked','Pclass_Sex','Age_Fare']
    num_feats = ['FamilySize','Age','Fare','Pclass_Sex','Age_Fare']
    ct, dim = build_preprocessor(df0, cat_feats, num_feats)

    manager = Manager()
    global_best = manager.Value('f', 0.0)
    lock = manager.Lock()

    exps = [10000] * 1
    q = Queue()
    procs = []
    for ex in exps:
        p = Process(target=run_experiment,
                    args=(ex, ct, dim, cat_feats, num_feats, global_best, lock, q))
        p.start()
        procs.append(p)
    results = [q.get() for _ in exps]
    for p in procs:
        p.join()
    best_epochs, best_val = max(results, key=lambda x: x[1])
    print(f"Best config: {best_epochs} epochs with val_acc={best_val:.4f}")

    # Final submission
    df_te = preprocess(pd.read_csv('test.csv'))
    Xte = ct.transform(df_te)
    if hasattr(Xte, 'toarray'):
        Xte = Xte.toarray()
    model = make_model(dim).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    with torch.no_grad():
        preds = (model(torch.tensor(Xte, dtype=torch.float32).to(device))
                 .squeeze() > 0.5).int().cpu().numpy()
    out_dir = os.path.join('submissions', 'submission_pt')
    os.makedirs(out_dir, exist_ok=True)
    submission = pd.DataFrame({'PassengerId': df_te['PassengerId'], 'Survived': preds})
    base = 'submission_pt.csv'
    fname, idx = base, 2
    while os.path.exists(os.path.join(out_dir, fname)):
        fname = f'submission_pt_{idx}.csv'
        idx += 1
    path = os.path.join(out_dir, fname)
    submission.to_csv(path, index=False)
    print(f"Submission saved to {path}")
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')
        print("Removed checkpoint best_model.pth")
