import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # 1.1 Ensure output directory exists
    output_dir = 'submission'
    os.makedirs(output_dir, exist_ok=True)

    # 2. Basic feature engineering
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)

    # 3. Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = train[features]
    y = train['Survived']

    # 4. Train/test split for local validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate on validation set
    accuracy = model.score(X_val, y_val)
    print(f"Local validation accuracy: {accuracy:.4f}")

    # 7. Retrain on all data & predict on test set
    model.fit(X, y)
    preds = model.predict(test[features])

    # 8. Save submission file
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': preds
    })
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"🏁 Written submission to {submission_path}")

if __name__ == '__main__':
    main()
