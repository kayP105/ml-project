import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score

def clean_and_balance_data(df):
    df = df[['text', 'label']].dropna()

    # Downsample majority class
    fake = df[df.label == 1]
    real = df[df.label == 0]
    min_len = min(len(fake), len(real))

    fake_balanced = resample(fake, replace=False, n_samples=min_len, random_state=42)
    real_balanced = resample(real, replace=False, n_samples=min_len, random_state=42)
    df_balanced = pd.concat([fake_balanced, real_balanced]).sample(frac=1, random_state=42)

    return df_balanced

def train_and_save_model():
    df = pd.read_csv('data/fake_news_dataset.csv')
    df = clean_and_balance_data(df)

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        ngram_range=(1, 2),
        min_df=3
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    os.makedirs('model', exist_ok=True)
    pickle.dump(model, open('model/fake_news_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

    return acc

if __name__ == '__main__':
    acc = train_and_save_model()
    print(f"✅ Training complete. Accuracy: {acc:.2%}")
