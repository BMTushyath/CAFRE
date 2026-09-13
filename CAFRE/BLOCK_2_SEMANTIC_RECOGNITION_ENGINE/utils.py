"""
utils.py — Shared Utilities for SRE
====================================
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

def detect_column_dtype(series: pd.Series) -> str:
    cleaned = series.dropna()
    if len(cleaned) == 0: return "unknown"
    if cleaned.nunique() == 2: return "binary"
    if pd.api.types.is_numeric_dtype(cleaned):
        if cleaned.nunique() <= 20: return "categorical"
        return "numeric"
    return "categorical"

def compute_entropy(series: pd.Series) -> float:
    cleaned = series.dropna()
    if len(cleaned) == 0: return 0.0
    probs = cleaned.value_counts(normalize=True).values
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def encode_categorical(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype("category").cat.codes.replace(-1, np.nan)

def print_header(title: str, char: str = "=", width: int = 70) -> None:
    print(char * width)
    print(f"  {title}")
    print(char * width)
