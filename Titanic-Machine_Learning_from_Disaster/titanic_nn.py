import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess(df):
    """
    Enhanced feature engineering for Titanic dataset.
    """
    df = df.copy()
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # Fill missing Age and Fare
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Rare')
    title_map = {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Rare':4}
    df['Title'] = df['Title'].map(title_map).astype(int)
    # Embarked encoding
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    return df


def main():
    print("Loading and preprocessing data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train = preprocess(train)
    test = preprocess(test)

    feature_cols = ['Pclass','Sex','Age','Fare','FamilySize','Title','Embarked']
    X = train[feature_cols]
    y = train['Survived']
    X_test = test[feature_cols]

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Preprocessing pipelines: scale numerical, encode categorical
    numeric_features = ['Age','Fare','FamilySize']
    categorical_features = ['Pclass','Sex','Title','Embarked']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Define MLP with early stopping and adaptive learning rate
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,64,32),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True,
        random_state=42
    )

    # Full pipeline
    model = Pipeline([
        ('preproc', preprocessor),
        ('mlp', mlp)
    ])

    print("Training neural network with pipeline...")
    model.fit(X_train, y_train)

    # Validation performance
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Predict on test set
    test_preds = model.predict(X_test)

    # Save submission
    out_dir = os.path.join('submissions', 'submission_nn')
    os.makedirs(out_dir, exist_ok=True)
    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_preds})
    base_name = 'submission_nn.csv'
    filename = base_name
    i = 2
    while os.path.exists(os.path.join(out_dir, filename)):
        filename = f"submission_nn_{i}.csv"
        i += 1
    submission_path = os.path.join(out_dir, filename)
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == '__main__':
    main()
