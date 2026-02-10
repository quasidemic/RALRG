from os.path import join
import json
#
import numpy as np
import pandas as pd

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_PATHS=[
    "/home/ubuntu/ragstuff/data/structured-search_mobility.csv",
    "/home/ubuntu/ragstuff/data/structured-search-preference.csv"
]

JSON_PATHS=[
    "/home/ubuntu/ragstuff/data/structured-search_mobility.json",
    "/home/ubuntu/ragstuff/data/structured-search-preference.json"
]

CSV_PATH = "/home/ubuntu/ragstuff/data/train-data.csv"

MODEL_PATH="./models/gte-large_abstracts"

# Load model
model=SetFitModel.from_pretrained(MODEL_PATH)

# Join data 
abstracts_df=pd.DataFrame()

for datapath in DATA_PATHS:
    df=pd.read_csv(datapath)

    abstracts_df=pd.concat([abstracts_df, df])

abstracts_df=abstracts_df.drop_duplicates(subset=['Url'])

# Join JSON
all_items=[]
urls_in=None

for jsonpath in JSON_PATHS:
    with open(jsonpath, "r") as f:
        record = json.load(f)
        items=record.get('items')

    if urls_in:
        items_add=[item for item in items if item.get('url') not in urls_in]
        all_items=all_items + items_add
    else:
        all_items=all_items + items

    urls_in=[item.get('url') for item in all_items]

# Predict function

def predict_with_threshold(
    texts,
    model=model,
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
    model=model,
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

# Predict

abstracts_predict_df=add_setfit_predictions(
    abstracts_df,
    model,
    threshold=0.8
)

# Store as csv
abstracts_predict_df.to_csv(join("/home/ubuntu/ragstuff/output", "articles_predicted.csv"), index=False)

# Select articles and store JSON
urls_keep = abstracts_predict_df.loc[abstracts_predict_df['predict_cat'].isin(['relevant']), 'Url'].to_list()

items_keep=[item for item in all_items if item.get('url') in urls_keep]

# fix attachments (has to be either url or path)
for item in items_keep:                          # items = your list of dicts
    for att in item.get("attachments", []):
        att.pop("url", None)

record_out = {
    'config': record.get('config'),
    'version': record.get('version'),
    'collections': record.get('collections'),
    'items': [item for item in items_keep if len(item.get('attachments')) > 0]
}

with open(join('/home/ubuntu/ragstuff/output', 'items_import.json'), 'w') as f:
    json.dump(record_out, f)
