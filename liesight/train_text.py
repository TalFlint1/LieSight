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

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = tokenize_data(tokenizer, df_train["statement"].tolist(), df_train["label"].tolist())
    val_dataset = tokenize_data(tokenizer, df_val["statement"].tolist(), df_val["label"].tolist())

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./models/roberta-text",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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
    trainer.save_model("./models/roberta-text")

if __name__ == "__main__":
    main()
