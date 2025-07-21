# titanic_pt_evolutionary.py
"""Pure evolutionary search with verbose logging.
Each generation uses a *new* random 80/20 validation split shared by **all** individuals,
so fitness is comparable while still rotating through the dataset to reduce over‑fitting.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from multiprocessing import Pool

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Titanic EA — rotating 80/20 splits')
    p.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    p.add_argument('--pop', type=int, default=1000, help='population size')
    p.add_argument('--gens', type=int, default=30, help='number of generations')
    p.add_argument('--mut', type=float, default=0.5, help='mutation std‑dev for weights')
    p.add_argument('--fresh', type=float, default=0.30, help='fraction of random newcomers each generation')
    return p.parse_args()

args = parse_args()

def get_device():
    if args.device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

device = get_device()

# --------------------------- Data ---------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
         'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
                                   'Rare': 4}).fillna(4).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Deck'] = df['Cabin'].fillna('U').str[0].map(
        {d: i for i, d in enumerate('ABCDEFGU')}).fillna(6).astype(int)
    df['TicketPrefix'] = (
        df['Ticket']
        .str.replace(r'\d+', '', regex=True)
        .str.replace(r'[./]', '', regex=True)
        .str.split().str[0]
        .replace('', 'NONE')
    )
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['AgeBin'] = pd.qcut(df['Age'], 4, labels=False)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    df['Pclass_Sex'] = df['Pclass'] * df['Sex']
    df['Age_Fare'] = df['Age'] * df['Fare']
    return df

CAT_COLS = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'FamilySize', 'Title',
            'Deck', 'TicketPrefix', 'Embarked', 'Pclass_Sex', 'Age_Fare']
NUM_COLS = ['FamilySize', 'Age', 'Fare', 'Pclass_Sex', 'Age_Fare']

# --------------------------- Model ---------------------------

def build_preprocessor(df):
    ct = ColumnTransformer([
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ])
    ct.fit(df)
    return ct, ct.transform(df).shape[1]

def make_model(dim: int):
    return nn.Sequential(
        nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1), nn.Sigmoid(),
    )

# --------------------------- EA Ops ---------------------------

def accuracy(state, X_val, y_val, dim):
    model = make_model(dim).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds = (model(Xv).squeeze() > 0.5).cpu().numpy().astype(int)
    return np.mean(preds == y_val)

def eval_worker(args):
    state, X_val, y_val, dim = args
    return state, accuracy(state, X_val, y_val, dim)

def crossover(p1, p2):
    a = np.random.rand()
    return {k: a * p1[k] + (1 - a) * p2[k] for k in p1}

def mutate(state):
    return {k: v + torch.randn_like(v) * args.mut for k, v in state.items()}

# --------------------------- Main ---------------------------

if __name__ == '__main__':
    # prepare data
    df_train = preprocess(pd.read_csv('train.csv'))
    preproc, DIM = build_preprocessor(df_train)
    X_all = preproc.transform(df_train)
    X_all = X_all.toarray() if hasattr(X_all, 'toarray') else X_all
    y_all = df_train['Survived'].values

    # initialise population
    population = [{k: v.cpu() for k, v in make_model(DIM).to(device).state_dict().items()}
                  for _ in range(args.pop)]

    best_state, best_score = None, 0.0
    pool = Pool()

    for gen in range(1, args.gens + 1):
        print(f"\n=== Generation {gen}/{args.gens} ===")
        # new shared 80/20 split
        _, val_idx = train_test_split(np.arange(len(y_all)), test_size=0.2)
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # evaluation
        results = pool.map(
            eval_worker, [(ind, X_val, y_val, DIM) for ind in population])
        results.sort(key=lambda x: x[1], reverse=True)

        scores = [s for _, s in results]
        for idx, (_, sc) in enumerate(results):
            print(f"  Ind {idx:03d} : acc = {sc:.4f}")
        print(f"Summary → max:{max(scores):.4f}  min:{min(scores):.4f}  "
              f"mean:{np.mean(scores):.4f}  std:{np.std(scores):.4f}")

        if scores[0] > best_score:
            best_state, best_score = results[0]
        print(f"Current best (all gens) : {best_score:.4f}\n")

        # evolve
        n_parents = int(args.pop * 0.2)
        parents = [s for s, _ in results[:n_parents]]
        new_pop = parents.copy()

        n_fresh = int(args.pop * args.fresh)
        for _ in range(n_fresh):
            new_pop.append({k: v.cpu() for k, v in make_model(DIM).to(device).state_dict().items()})

        while len(new_pop) < args.pop:
            p1, p2 = np.random.choice(parents, 2, replace=False)
            new_pop.append(mutate(crossover(p1, p2)))

        population = new_pop

    print(f"\n=== Finished ===\nBest validation accuracy: {best_score:.4f}")

    # submission
    df_test = preprocess(pd.read_csv('test.csv'))
    X_test = preproc.transform(df_test)
    X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

    final_model = make_model(DIM).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()
    with torch.no_grad():
        preds = (final_model(torch.tensor(X_test, dtype=torch.float32).to(device))
                 .squeeze() > 0.5).int().cpu().numpy()

    out_dir = os.path.join('submissions', 'submission_evo')
    os.makedirs(out_dir, exist_ok=True)
    n_existing = sum(1 for f in os.listdir(out_dir)
                     if f.startswith('submission_evo_') and f.endswith('.csv')) + 1
    fname = f'submission_evo_{n_existing}.csv'
    pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': preds})\
        .to_csv(os.path.join(out_dir, fname), index=False)
    print(f"Submission written → {os.path.join(out_dir, fname)}")
