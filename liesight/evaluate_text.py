# evaluate_text.py

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from liesight.data_loader import load_liar_dataset
from datasets import Dataset
import numpy as np

def tokenize_data(tokenizer, texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    encodings['labels'] = labels
    return Dataset.from_dict(encodings)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    print("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained("./models/roberta-large")
    model = RobertaForSequenceClassification.from_pretrained("./models/roberta-large")

    print("Loading test data...")
    df_test = load_liar_dataset("test")
    test_dataset = tokenize_data(tokenizer, df_test["statement"].tolist(), df_test["label"].tolist())

    print("Running evaluation...")
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(test_dataset)
    print("Test set evaluation:")
    print(results)

if __name__ == "__main__":
    main()
