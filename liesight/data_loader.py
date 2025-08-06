import pandas as pd
from pathlib import Path

LABEL_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}

# Dynamically resolve project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

def load_liar_dataset(split="train"):
    path = DATA_DIR / f"{split}.tsv"
    print(f"Resolved path: {path}")  # Add this line
    df = pd.read_csv(path, sep="\t", header=None, quoting=3)
    df = df[[2, 1]]  # column 1 = statement, column 2 = label
    df.columns = ["statement", "label"]
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna()
    return df
