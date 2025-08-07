import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from liesight.data_loader import load_liar_dataset
from datasets import Dataset
import numpy as np
import os

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
    print("Loading data...")
    df_train = load_liar_dataset("train")
    df_val = load_liar_dataset("valid")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    train_dataset = tokenize_data(tokenizer, df_train["statement"].tolist(), df_train["label"].tolist())
    val_dataset = tokenize_data(tokenizer, df_val["statement"].tolist(), df_val["label"].tolist())

    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./models/roberta-large",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model("./models/roberta-large")

if __name__ == "__main__":
    main()
