import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Suppress warnings from LightGBM and XGBoost
warnings.filterwarnings('ignore', category=UserWarning)


def preprocess(df):
    """
    Enhanced feature engineering: titles, family size, deck, ticket prefix,
    fare/age bins.
    """
    df = df.copy()
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    title_map = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}
    df['Title'] = df['Title'].map(title_map).fillna(title_map['Rare']).astype(int)

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Deck from cabin
    df['Deck'] = df['Cabin'].fillna('U').str[0]
    deck_map = {d:i for i,d in enumerate(list('ABCDEFGU'))}
    df['Deck'] = df['Deck'].map(deck_map).fillna(deck_map['U']).astype(int)

    # Ticket prefix
    df['TicketPrefix'] = (
        df['Ticket']
        .str.replace(r'\d', '', regex=True)
        .str.replace(r'\.', '', regex=True)
        .str.replace('/', '', regex=True)
        .str.strip()
        .str.split()
        .str[0]
    )
    df['TicketPrefix'] = df['TicketPrefix'].replace('', 'NONE')

    # Fill numeric columns
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    return df


def main():
    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Feature engineering
    train = preprocess(train)
    test = preprocess(test)

    # Define feature sets
    numeric_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Title', 'Deck', 'Embarked', 'TicketPrefix']

    # Preprocessing pipelines
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Base learners with suppressed verbosity
    estimators = [
        ('lgbm', lgb.LGBMClassifier(random_state=42, n_estimators=200, num_leaves=31, verbosity=-1)),
        ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=200, verbosity=0, use_label_encoder=False)),
        ('cat', CatBoostClassifier(random_state=42, verbose=0, iterations=200))
    ]

    # Stacking classifier
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )

    # Full pipeline
    clf = Pipeline([
        ('preproc', preprocessor),
        ('stack', stack)
    ])

    # Prepare data for validation
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
    X = train.drop(drop_cols, axis=1)
    y = train['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Stacking CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Fit on train and evaluate hold-out
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")

    # Retrain on full data and predict test
    clf.fit(X, y)
    X_test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    preds = clf.predict(X_test)

    # Save submission
    out_dir = 'submission_ensemble'
    os.makedirs(out_dir, exist_ok=True)
    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
    file_path = os.path.join(out_dir, 'submission_ensemble.csv')
    submission.to_csv(file_path, index=False)
    print(f"Ensemble submission saved to {file_path}")

if __name__ == '__main__':
    main()
