import os
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Optionally suppress irrelevant warnings
warnings.filterwarnings("ignore", category=UserWarning)


def preprocess(df):
    """
    Apply feature engineering to the DataFrame.
    """
    print("  - Encoding 'Sex' and filling missing 'Age' and 'Fare'...")
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    print("  - Creating 'FamilySize' feature and extracting 'Title'...")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    titles = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    titles = titles.replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
         'Sir', 'Jonkheer', 'Dona'], 'Rare'
    )
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = titles.map(title_mapping).fillna(title_mapping['Rare']).astype(int)

    print("  - Encoding 'Embarked'...")
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    print("  - Feature engineering complete.")
    return df


def main():
    print("Loading data files...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    output_dir = 'submission_lgbm'
    print(f"Ensuring output directory '{output_dir}' exists...")
    os.makedirs(output_dir, exist_ok=True)

    print("Preprocessing training data:")
    train = preprocess(train)
    print("Preprocessing test data:")
    test = preprocess(test)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title', 'Embarked']
    print(f"Using features: {features}")
    X = train[features]
    y = train['Survived']

    print("Splitting data into train/validation sets (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Setting up LightGBM classifier and hyperparameter grid...")
    lgb_clf = lgb.LGBMClassifier(
        random_state=42,
        verbosity=-1,
        force_row_wise=True
    )
    param_grid = {
        'num_leaves': [31, 50, 70],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.1, 0.05],
        'n_estimators': [100, 200, 300],
        'min_child_samples': [5, 10, 20]
    }

    print("Initializing GridSearchCV (verbose=2) for hyperparameter tuning...")
    grid = GridSearchCV(
        estimator=lgb_clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )

    print("Running GridSearchCV...")
    grid.fit(X_train, y_train)
    print("GridSearchCV complete.")
    print("Best LGBM Params:", grid.best_params_)
    print("Tuned CV accuracy:", grid.best_score_)

    print("Validating best model on hold-out set...")
    best_model = grid.best_estimator_
    val_preds = best_model.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, val_preds))

    print("Retraining best model on full dataset and predicting on test set...")
    best_model.fit(X, y)
    test_preds = best_model.predict(test[features])

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_preds
    })
    submission_path = os.path.join(output_dir, 'submission_lgbm.csv')
    print(f"Saving submission to {submission_path}...")
    submission.to_csv(submission_path, index=False)
    print("🏁 Submission saved successfully.")

if __name__ == '__main__':
    main()