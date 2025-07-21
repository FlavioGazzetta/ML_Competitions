import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from tqdm import tqdm


def preprocess(df):
    """
    Basic feature engineering for Titanic dataset.
    """
    df = df.copy()
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # Fill Age and Fare
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = df['Title'].map(title_map).astype(int)
    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    return df


def main():
    # Load datasets
    print("Loading train and test data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Preprocess
    print("Preprocessing data...")
    train = preprocess(train)
    test = preprocess(test)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title', 'Embarked']
    X = train[features].values
    y = train['Survived'].values
    X_test = test[features].values

    # KFold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    print(f"Training with {n_splits}-fold Stratified CV...")
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc='Folds')):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # LightGBM dataset
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)

        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': 42
        }

        print("Training LightGBM...")
        # Use callbacks for early stopping and logging
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # OOF predictions
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        # Test predictions
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

        fold_acc = accuracy_score(y_val, (oof_preds[val_idx] > 0.5).astype(int))
        print(f"Fold {fold+1} accuracy: {fold_acc:.4f}")

    # Overall CV
    cv_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"\nOverall CV accuracy: {cv_acc:.4f}")

    out_dir = os.path.join('submissions', 'submission_kfold')
    os.makedirs(out_dir, exist_ok=True)
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': (test_preds > 0.5).astype(int)
    })
    # Determine submission filename without overwriting
    base_name = 'submission_kfold_lgbm.csv'
    filename = base_name
    i = 2
    while os.path.exists(os.path.join(out_dir, filename)):
        filename = f"submission_kfold_lgbm_{i}.csv"
        i += 1
    submission_path = os.path.join(out_dir, filename)
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == '__main__':
    main()
