import os 
from os.path import join
import json
from pathlib import Path
#
import numpy as np
import pandas as pd

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

# Predict function

def predict_with_threshold(
    texts,
    model,
    threshold: float = 0.5,
    ooc_label: str = "OOC",
):
    """
    Predict labels for a list/Series of texts, using a probability threshold.

    Parameters
    ----------
    model : SetFitModel
        Trained SetFit model.
    texts : iterable of str
        Texts to classify.
    threshold : float
        Minimum max-class probability required to assign a label.
        If the best class prob < threshold, assign `ooc_label`.
    ooc_label : str
        Label to use for "out-of-category" predictions.

    Returns
    -------
    labels : list of str
        Final labels per text (including `ooc_label` where applicable).
    max_probs : np.ndarray
        Max probability per text.
    """

    # Get probabilities: shape (n_samples, n_classes)
    probs = model.predict_proba(texts)
    probs = np.asarray(probs)

    max_probs = probs.max(axis=1)
    max_idx = probs.argmax(axis=1)

    # Get class labels in correct order
    classes = list(model.labels)

    labels = []
    for p, idx in zip(max_probs, max_idx):
        if p >= threshold:
            labels.append(classes[idx])
        else:
            labels.append(ooc_label)

    return labels, max_probs

# Add predictions function

def add_setfit_predictions(
    df: pd.DataFrame,
    model,
    text_col: str = "Abstract Note",
    threshold: float = 0.5,
    ooc_label: str = "OOC",
    predict_col: str = "predict_cat",
    probs_col: str = "predict_prob"
):
    """
    Add SetFit predictions to a DataFrame as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame; must contain `text_col`.
    model_path : str
        Local directory or Hub id for the trained SetFit model.
    text_col : str
        Name of the text column (default: "Abstract note").
    threshold : float
        Threshold on max class probability to assign a label.
    ooc_label : str
        Label to use when max prob < threshold.
    new_col : str
        Name of the new prediction column (default: "predict_cat").

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with an added `new_col`.
    """
    if text_col not in df.columns:
        raise ValueError(f"DataFrame is missing required text column: {text_col!r}")

    df_out = df.copy()

    # Predict only on non-null texts
    mask = df_out[text_col].notna()
    texts = df_out.loc[mask, text_col].astype(str).to_list()

    labels, probs = predict_with_threshold(
        model=model,
        texts=texts,
        threshold=threshold,
        ooc_label=ooc_label,
    )

    df_out[predict_col] = ooc_label  # default
    df_out.loc[mask, predict_col] = labels
    df_out.loc[mask, probs_col] = probs

    return df_out