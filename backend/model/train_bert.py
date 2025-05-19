import os
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

def verify_environment():
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    assert torch.__version__.startswith("2.2"), "Requires PyTorch 2.2.x"
    assert np.__version__.startswith("1.26"), "Requires NumPy 1.26.x"

def load_data():
    df = pd.read_csv('data/fake_news_dataset.csv')
    df = df[['text', 'label']].dropna()
    return df
def prepare_datasets(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def train_and_save_model():
    verify_environment()
    
    # Load data
    df = load_data()
    train_dataset, test_dataset = prepare_datasets(df)

    # Initialize model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Training setup
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        remove_unused_columns=False  # Important fix
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train and save
    trainer.train()
    model.save_pretrained('model/distilbert-fakenews')
    tokenizer.save_pretrained('model/distilbert-fakenews')

if __name__ == '__main__':
    train_and_save_model()
