# titanic_pt_evolutionary_hold.py
"""
Evolutionary search (PyTorch) for the Titanic-survival task **with**

1. A permanent stratified hold-out set (default 15 %) never touched during
   evolution.
2. Average fitness over *n* random 80 / 20 validation splits each generation
   (`--splits`, default = 3).  Every genome sees exactly the same splits per
   generation, so fitness is comparable while smoothing lucky-split variance.
3. Optional relative-comparison reward (`--cmp`).  Set `--cmp 0` to disable.

Example:
    python titanic_pt_evolutionary_hold.py --pop 300 --gens 120 \
           --splits 3 --cmp 0.1 --hold 0.15 --device cuda
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from multiprocessing import Pool
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Titanic EA — permanent hold-out + averaged multi-split fitness")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--pop", type=int, default=2000, help="population size")
    p.add_argument("--gens", type=int, default=100, help="number of generations")
    p.add_argument("--mut", type=float, default=1.0, help="mutation std-dev for weights")
    p.add_argument("--fresh", type=float, default=0.10,
                   help="fraction of random newcomers each generation")
    p.add_argument("--cmp", type=float, default=0.2,
                   help="weight of relative-comparison reward (0 = disabled)")
    p.add_argument("--hold", type=float, default=0.1,
                   help="fraction of data kept as an untouched hold-out set")
    p.add_argument("--splits", type=int, default=5,
                   help="number of random 80/20 splits averaged per generation")
    return p.parse_args()

args = parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

device = get_device()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle-style feature engineering."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
         "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    df["Title"] = df["Title"].map(
        {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    ).fillna(4).astype(int)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Deck"] = df["Cabin"].fillna("U").str[0].map(
        {d: i for i, d in enumerate("ABCDEFGU")}
    ).fillna(6).astype(int)

    df["TicketPrefix"] = (
        df["Ticket"]
          .str.replace(r"\d+", "", regex=True)
          .str.replace(r"[./]", "", regex=True)
          .str.split().str[0]
          .replace("", "NONE")
    )

    df["Age"]  = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["AgeBin"]  = pd.qcut(df["Age"],  4, labels=False)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)

    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2}).astype(int)

    df["Pclass_Sex"] = df["Pclass"] * df["Sex"]
    df["Age_Fare"]   = df["Age"] * df["Fare"]
    return df


CAT_COLS = ["Pclass", "Sex", "AgeBin", "FareBin", "FamilySize", "Title",
            "Deck", "TicketPrefix", "Embarked", "Pclass_Sex", "Age_Fare"]
NUM_COLS = ["FamilySize", "Age", "Fare", "Pclass_Sex", "Age_Fare"]


def build_preprocessor(df):
    ct = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    ct.fit(df)
    return ct, ct.transform(df).shape[1]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def make_model(dim: int):
    return nn.Sequential(
        nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1), nn.Sigmoid(),
    )


def accuracy(state, X_val, y_val, dim):
    """Binary-accuracy of `state` on (X_val, y_val)."""
    model = make_model(dim).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds = (model(Xv).squeeze() > 0.5).cpu().numpy().astype(int)
    return float(np.mean(preds == y_val))


def multi_split_accuracy(state, X, y, dim, val_idx_sets):
    """Mean accuracy over `val_idx_sets` splits."""
    return float(np.mean([accuracy(state, X[idx], y[idx], dim) for idx in val_idx_sets]))


# ---------------------------------------------------------------------------
# Evolutionary operators
# ---------------------------------------------------------------------------

def mutate(state):
    return {k: v + torch.randn_like(v) * args.mut for k, v in state.items()}


def crossover(p1, p2):
    a = np.random.rand()
    return {k: a * p1[k] + (1.0 - a) * p2[k] for k in p1}


def eval_worker(payload):
    state, X, y, dim, val_idx_sets = payload
    return state, multi_split_accuracy(state, X, y, dim, val_idx_sets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 1  Load data and carve a permanent hold-out
    # ----------------------------------------------------------------------
    df_train = preprocess(pd.read_csv("train.csv"))
    preproc, DIM = build_preprocessor(df_train)
    X_all = preproc.transform(df_train)
    X_all = X_all.toarray() if hasattr(X_all, "toarray") else X_all
    y_all = df_train["Survived"].values

    train_idx, hold_idx = train_test_split(
        np.arange(len(y_all)), test_size=args.hold,
        stratify=y_all, random_state=42)

    X_train_all, y_train_all = X_all[train_idx], y_all[train_idx]
    X_hold,        y_hold    = X_all[hold_idx],  y_all[hold_idx]

    print(f"Training rows: {len(train_idx)}   |   Hold-out rows: {len(hold_idx)}")

    # ----------------------------------------------------------------------
    # 2  Initial population
    # ----------------------------------------------------------------------
    population = [
        {k: v.cpu() for k, v in make_model(DIM).to(device).state_dict().items()}
        for _ in range(args.pop)
    ]

    best_state, best_val_acc = None, 0.0     # running best (moving-split score)
    best_hold_acc            = 0.0           # running best on true hold-out
    pool = Pool()

    # ----------------------------------------------------------------------
    # 3  Evolution loop
    # ----------------------------------------------------------------------
    for gen in range(1, args.gens + 1):
        print(f"\n=== Generation {gen}/{args.gens} ===")

        # Pre-generate the SAME validation splits for the whole population
        val_idx_sets = [
            train_test_split(np.arange(len(y_train_all)), test_size=0.2)[1]
            for _ in range(args.splits)
        ]

        # --- Evaluation (mean accuracy) -----------------------------------
        tasks = [
            (ind, X_train_all, y_train_all, DIM, val_idx_sets)
            for ind in population
        ]
        results = pool.map(eval_worker, tasks)
        states, accs = zip(*results)

        # --- Relative-comparison reward -----------------------------------
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs) + 1e-12)
        fitnesses = [
            acc + args.cmp * ((acc - mean_acc) / std_acc)  # z-score bonus
            for acc in accs
        ]

        ranked = sorted(
            zip(states, accs, fitnesses), key=lambda t: t[2], reverse=True)

        states_sorted, accs_sorted, _ = zip(*ranked)
        print(f"Summary → max:{max(accs):.4f}  min:{min(accs):.4f}  "
              f"mean:{mean_acc:.4f}  std:{std_acc:.4f}")

        # Track best validation accuracy (moving splits)
        if accs_sorted[0] > best_val_acc:
            best_val_acc = accs_sorted[0]
            best_state   = states_sorted[0]

        # Evaluate current leader on hold-out for curiosity / early stop
        hold_acc_gen = accuracy(states_sorted[0], X_hold, y_hold, DIM)
        best_hold_acc = max(best_hold_acc, hold_acc_gen)
        print(f"Leader hold-out acc : {hold_acc_gen:.4f}   "
              f"(best ever: {best_hold_acc:.4f})")

        # --- Selection ----------------------------------------------------
        n_parents = int(args.pop * 0.2)
        parents   = list(states_sorted[:n_parents])

        # Elitism: carry parents forward untouched
        new_pop = parents.copy()

        # Fresh random genomes
        n_fresh = int(args.pop * args.fresh)
        for _ in range(n_fresh):
            new_pop.append(
                {k: v.cpu()
                 for k, v in make_model(DIM).to(device).state_dict().items()})

        # Fill up with crossover + mutation
        while len(new_pop) < args.pop:
            p1, p2 = np.random.choice(parents, 2, replace=False)
            child  = mutate(crossover(p1, p2))
            new_pop.append(child)

        population = new_pop

    # ----------------------------------------------------------------------
    # 4  Finished — evaluate best overall on hold-out
    # ----------------------------------------------------------------------
    final_hold_acc = accuracy(best_state, X_hold, y_hold, DIM)
    print("\n=== Finished ===")
    print(f"Best moving-split validation accuracy : {best_val_acc:.4f}")
    print(f"Hold-out accuracy                      : {final_hold_acc:.4f}")

    # ----------------------------------------------------------------------
    # 5  Make Kaggle-style submission
    # ----------------------------------------------------------------------
    df_test = preprocess(pd.read_csv("test.csv"))
    X_test  = preproc.transform(df_test)
    X_test  = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    final_model = make_model(DIM).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()
    with torch.no_grad():
        preds = (
            final_model(torch.tensor(X_test, dtype=torch.float32).to(device))
            .squeeze() > 0.5
        ).int().cpu().numpy()

    out_dir = os.path.join("submissions", "submission_evo")
    os.makedirs(out_dir, exist_ok=True)
    run = sum(f.startswith("submission_evo_") for f in os.listdir(out_dir)) + 1
    fname = f"submission_evo_{run}.csv"
    pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": preds})\
        .to_csv(os.path.join(out_dir, fname), index=False)
    print(f"Submission written → {os.path.join(out_dir, fname)}")
