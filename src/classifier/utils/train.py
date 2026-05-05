import os
from os.path import join
import json
from pathlib import Path
import random
import uuid

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pysbd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TEXT_COLUMN = "Abstract Note"
LABEL_COLUMN = "label"
KEY_COLUMN = "Key"
CATEGORY_COLUMN = "category"
TEXT_FIELD = "text"


def synthesize_other_texts(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Upscale 'other' label by creating synthetic texts composed of random segments
    from existing 'other' samples. Each synthetic text uses the mean sentence count
    of the existing 'other' texts as its length.
    """
    other_df = df[df[LABEL_COLUMN] == "other"]
    relevant_count = len(df[df[LABEL_COLUMN] == "relevant"])
    other_count = len(other_df)

    # Only generate when we have fewer 'other' samples than 'relevant' samples
    samples_needed = relevant_count - other_count
    if samples_needed <= 0 or other_df.empty:
        return df

    segmenter = pysbd.Segmenter(language="en", clean=True)
    sentence_pool = []
    sentence_counts = []
    for text in other_df[TEXT_FIELD]:
        segments = segmenter.segment(text)
        if segments:
            sentence_pool.extend(segments)
            sentence_counts.append(len(segments))

    if not sentence_pool or not sentence_counts:
        return df

    mean_sentences = max(1, round(sum(sentence_counts) / len(sentence_counts)))
    rng = random.Random(seed)
    synthetic_rows = []

    for _ in range(samples_needed):
        chosen_segments = [rng.choice(sentence_pool) for _ in range(mean_sentences)]
        synthetic_rows.append(
            {
                KEY_COLUMN: f"synthetic-other-{uuid.uuid4().hex[:8]}",
                TEXT_FIELD: " ".join(chosen_segments),
                LABEL_COLUMN: "other",
            }
        )

    synthetic_df = pd.DataFrame(synthetic_rows, columns=[KEY_COLUMN, TEXT_FIELD, LABEL_COLUMN])
    return pd.concat([df, synthetic_df], ignore_index=True)