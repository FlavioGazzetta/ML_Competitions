import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF logs
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    print("TensorFlow is not installed. Please install it with `pip install tensorflow`")
    exit(1)


def preprocess(df):
    """
    Enhanced feature engineering: titles, family size, deck, ticket prefix, binned age/fare.
    """
    df = df.copy()
    # Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    title_map = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}
    df['Title'] = df['Title'].map(title_map).fillna(4).astype(int)
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # Deck
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
    # Numeric fill and binning
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['AgeBin'] = pd.qcut(df['Age'], 4, labels=False)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    # Sex
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    return df


def build_model(input_dim):
    """
    Build a Keras model with functional API, batch norm, dropout, and LR scheduler.
    """
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    print("Loading and preprocessing data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train = preprocess(train)
    test = preprocess(test)

    # Features to use
    feature_cols = ['Pclass','Sex','AgeBin','FareBin','FamilySize','Title','Deck','TicketPrefix','Embarked']
    X = train[feature_cols]
    y = train['Survived']
    X_test = test[feature_cols]

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Preprocessing pipeline
    numeric_feats = ['FamilySize']
    cat_feats = ['Pclass','Sex','AgeBin','FareBin','Title','Deck','TicketPrefix','Embarked']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    # Build and train model
    model = build_model(X_train_proc.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    ]
    print("Training TensorFlow model...")
    model.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=300,
        batch_size=16,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val_proc, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Prepare submission
    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': (model.predict(X_test_proc) > 0.5).astype(int).reshape(-1)})
    # Ensure correct format and order
    submission = submission[['PassengerId','Survived']]
    submission = submission.sort_values('PassengerId')
    assert submission.shape[0] == len(test), "Submission row count mismatch."

    # Save without overwriting
    out_dir = os.path.join('submissions', 'submission_tf')
    os.makedirs(out_dir, exist_ok=True)
    base = 'submission_tf.csv'
    fname = base
    i = 2
    while os.path.exists(os.path.join(out_dir, fname)):
        fname = f"submission_tf_{i}.csv"
        i += 1
    path = os.path.join(out_dir, fname)
    submission.to_csv(path, index=False)
    print(f"Submission saved to {path}")

if __name__ == '__main__':
    main()
