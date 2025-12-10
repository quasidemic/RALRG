import os
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
CSV_PATH = "/home/ubuntu/ragstuff/data/train-data.csv"

TEXT_COLUMN = "Abstract Note"
LABEL_COLUMN = "label"
KEY_COLUMN = "Key"
CATEGORY_COLUMN = "category"

# Hugging Face embedding model
EMBEDDING_MODEL_ID = "thenlper/gte-large"

# Output directory for the trained model
OUTPUT_DIR = "./models/gte-large_abstracts"

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


def main(seed_use=1764933039): # Unix Epoch 2025-12-05 12:10
    # -------------------------------------------------------------------
    # 1. Read CSV into DataFrame
    # -------------------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Basic checks
    required_cols = {KEY_COLUMN, TEXT_COLUMN, CATEGORY_COLUMN}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # -------------------------------------------------------------------
    # 2. Create 'label' column based on 'category'
    #    Keep 'segregation' and 'preference', everything else -> 'other'
    # -------------------------------------------------------------------
    df[LABEL_COLUMN] = df[CATEGORY_COLUMN].apply(
        lambda c: "relevant" if c in ["segregation", "preference"] else "other"
    )

    # -------------------------------------------------------------------
    # 3. Keep only columns key, abstract note, label
    # -------------------------------------------------------------------
    df = df[[KEY_COLUMN, TEXT_COLUMN, LABEL_COLUMN]].copy()

    # Drop rows with missing text or label
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).rename(columns = {'Abstract Note': TEXT_FIELD})

    # -------------------------------------------------------------------
    # 3a. Upscale the 'other' class with synthetic texts built from random
    #     segments of existing 'other' texts
    # -------------------------------------------------------------------
    df = synthesize_other_texts(df, seed_use)

    # -------------------------------------------------------------------
    # 4. Split into training and test data
    # -------------------------------------------------------------------
    # First, split into full train/test
    train_df, test_df = train_test_split(
        df,
        test_size=0.5,
        stratify=df[LABEL_COLUMN],
        random_state=seed_use, 
    )

    # -------------------------------------------------------------------
    # Convert pandas DataFrames to Hugging Face Datasets
    # -------------------------------------------------------------------
    train_dataset = Dataset.from_pandas(
        train_df,
        preserve_index=False,
    )
    test_dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False,
    )


    # -------------------------------------------------------------------
    # 5. Train a SetFit few-shot model using gte-large
    # -------------------------------------------------------------------
    model = SetFitModel.from_pretrained(
        EMBEDDING_MODEL_ID
    )
    
    training_args = TrainingArguments(
        batch_size=8,
        num_epochs=5,
        body_learning_rate=2e-5,
        seed=seed_use,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        output_dir=OUTPUT_DIR
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        metric="accuracy"
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # -------------------------------------------------------------------
    # Evaluate on the test set
    # -------------------------------------------------------------------
    print("Evaluating on the test set...")
    preds = trainer.model.predict(test_dataset[TEXT_FIELD])
    print(classification_report(test_dataset[LABEL_COLUMN], preds))

    # -------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":

    main()
