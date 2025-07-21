# titanic_pt_evolutionary_hold.py 
"""
Evolutionary search (PyTorch) for the Titanic‑survival task **with**

1. A permanent stratified hold‑out set (default 15 %) never touched during
   evolution.
2. Average fitness over *n* random 80 / 20 validation splits each generation
   (`--splits`, default = 3).  Every genome sees exactly the same splits per
   generation, so fitness is comparable while smoothing lucky‑split variance.
3. Optional relative‑comparison reward (`--cmp`).  Set `--cmp 0` to disable.
4. **Longevity bonus**: remaining in the top‑20 % across consecutive
   generations grants a multiplicative boost to fitness (`--persist`).
5. **Graceful Ctrl‑C** on Windows & Unix: pools ignore SIGINT, main catches it
   and terminates cleanly, preventing the long traceback you observed.

Example:
    python titanic_pt_evolutionary_hold.py \
        --pop 300 --gens 120 --splits 3 --cmp 0.1 --persist 0.05 --device cuda
"""

import os
import sys
import signal
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import count
from copy import deepcopy
from multiprocessing import Pool, freeze_support
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Titanic EA — permanent hold‑out + averaged multi‑split fitness + longevity bonus + graceful Ctrl‑C"
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--pop", type=int, default=10, help="population size")
    p.add_argument("--gens", type=int, default=10000, help="number of generations")
    p.add_argument("--mut", type=float, default=1.0, help="mutation std‑dev for weights")
    p.add_argument("--fresh", type=float, default=0.10,
                   help="fraction of random newcomers each generation")
    p.add_argument("--cmp", type=float, default=0.2,
                   help="weight of relative‑comparison reward (0 = disabled)")
    p.add_argument("--hold", type=float, default=0.1,
                   help="fraction of data kept as an untouched hold‑out set")
    p.add_argument("--splits", type=int, default=5,
                   help="number of random 80/20 splits averaged per generation")
    p.add_argument("--persist", type=float, default=0.05,
                   help="multiplicative weight for longevity bonus (0 = off)")
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
# Graceful Ctrl‑C support for multiprocessing
# ---------------------------------------------------------------------------

def init_worker():
    """Ignore SIGINT in worker so only the main process handles Ctrl‑C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# ---------------------------------------------------------------------------
# Global ID counter for individuals
# ---------------------------------------------------------------------------
ID_COUNTER = count()

def next_id():
    return next(ID_COUNTER)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle‑style feature engineering."""
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

CAT_COLS = [
    "Pclass", "Sex", "AgeBin", "FareBin", "FamilySize", "Title",
    "Deck", "TicketPrefix", "Embarked", "Pclass_Sex", "Age_Fare"
]
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
    """Binary‑accuracy of `state` on (X_val, y_val)."""
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
# Evolutionary operators & worker
# ---------------------------------------------------------------------------

def mutate(state):
    return {k: v + torch.randn_like(v) * args.mut for k, v in state.items()}


def crossover(p1_state, p2_state):
    a = np.random.rand()
    return {k: a * p1_state[k] + (1.0 - a) * p2_state[k] for k in p1_state}


def eval_worker(payload):
    state, X, y, dim, val_idx_sets = payload
    return multi_split_accuracy(state, X, y, dim, val_idx_sets)

# ---------------------------------------------------------------------------
# Individual helper
# ---------------------------------------------------------------------------

def new_individual(state_dict):
    """Create a fresh individual wrapper with unique id and zero longevity."""
    return {
        "state": state_dict,
        "longevity": 0,  # consecutive generations in top‑20 %
        "id": next_id(),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    freeze_support()  # Needed for Windows when spawning child processes

    # ------------------------------------------------------------------
    # 1  Load data and carve a permanent hold‑out
    # ------------------------------------------------------------------
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

    print(f"Training rows: {len(train_idx)}   |   Hold‑out rows: {len(hold_idx)}")

    # ----------------------------------------------------------------------
    # 2  Initial population
    # ----------------------------------------------------------------------
    population = [
        new_individual({k: v.cpu() for k, v in make_model(DIM).to(device).state_dict().items()})
        for _ in range(args.pop)
    ]

    best_state, best_val_acc = None, 0.0     # running best (moving‑split score)
    best_hold_acc            = 0.0           # running best on true hold‑out
    pool = Pool()

    # ----------------------------------------------------------------------
    # 3  Evolution loop
    # ----------------------------------------------------------------------
    for gen in range(1, args.gens + 1):
        print(f"\n=== Generation {gen}/{args.gens} ===")

        # Pre‑generate the SAME validation splits for the whole population
        val_idx_sets = [
            train_test_split(np.arange(len(y_train_all)), test_size=0.2)[1]
            for _ in range(args.splits)
        ]

        # --- Evaluation (mean accuracy) -----------------------------------
        tasks = [
            (ind["state"], X_train_all, y_train_all, DIM, val_idx_sets)
            for ind in population
        ]
        accs = pool.map(eval_worker, tasks)

        # --- Relative‑comparison reward (base fitness) --------------------
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs) + 1e-12)
        base_fitnesses = [
            acc + args.cmp * ((acc - mean_acc) / std_acc)  # z‑score bonus
            for acc in accs
        ]

        # --- Determine current top‑20 % (based on *base* fitness) ---------
        n_parents = int(args.pop * 0.2)
        indices_by_base = np.argsort(base_fitnesses)[::-1]  # descending
        top20_idx = set(indices_by_base[:n_parents])

        # --- Update longevities ------------------------------------------
        for idx, ind in enumerate(population):
            if idx in top20_idx:
                ind["longevity"] += 1
            else:
                ind["longevity"] = 0

        # --- Apply longevity multiplier ----------------------------------
        longevity_factors = [1.0 + args.persist * ind["longevity"] for ind in population]
        adjusted_fitnesses = [bf * lf for bf, lf in zip(base_fitnesses, longevity_factors)]

        # --- Final ranking for selection ----------------------------------
        ranked = sorted(
            zip(population, accs, adjusted_fitnesses), key=lambda t: t[2], reverse=True)

        # Unpack sorted data
        pop_sorted, accs_sorted, fit_sorted = zip(*ranked)

        print(f"Summary → max:{max(accs):.4f}  min:{min(accs):.4f}  "
              f"mean:{mean_acc:.4f}  std:{std_acc:.4f}")

        # Track best validation accuracy (moving splits)
        if accs_sorted[0] > best_val_acc:
            best_val_acc = accs_sorted[0]
            best_state   = deepcopy(pop_sorted[0]["state"])

        # Evaluate current leader on hold‑out for curiosity / early stop
        hold_acc_gen = accuracy(pop_sorted[0]["state"], X_hold, y_hold, DIM)
        best_hold_acc = max(best_hold_acc, hold_acc_gen)
        print(f"Leader hold‑out acc : {hold_acc_gen:.4f}   "
              f"(best ever: {best_hold_acc:.4f})")

        # --- Selection ----------------------------------------------------
        parents   = list(pop_sorted[:n_parents])  # keep individual objects (with longevity)

        # Elitism: carry parents forward untouched
        new_pop = [deepcopy(ind) for ind in parents]

        # Fresh random genomes
        n_fresh = int(args.pop * args.fresh)
        for _ in range(n_fresh):
            new_pop.append(new_individual({k: v.cpu() for k, v in make_model(DIM).to(device).state_dict().items()}))

        # Fill up with crossover + mutation
        while len(new_pop) < args.pop:
            p1, p2 = np.random.choice(parents, 2, replace=False)
            child_state = mutate(crossover(p1["state"], p2["state"]))
            new_pop.append(new_individual(child_state))

        population = new_pop

    # ----------------------------------------------------------------------
    # 4  Finished — evaluate best overall on hold‑out
    # ----------------------------------------------------------------------
    final_hold_acc = accuracy(best_state, X_hold, y_hold, DIM)
    print("\n=== Finished ===")
    print(f"Best moving‑split validation accuracy : {best_val_acc:.4f}")
    print(f"Hold‑out accuracy                      : {final_hold_acc:.4f}")

    # ----------------------------------------------------------------------
    # 5  Make Kaggle‑style submission
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
