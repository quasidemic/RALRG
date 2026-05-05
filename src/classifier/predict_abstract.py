import os 
from os.path import join
import json
from pathlib import Path
#
import numpy as np
import pandas as pd

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from utils.predict import add_setfit_predictions

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_DIR = Path(os.environ["PROJECT_DIR"])

DATA_PATHS=[
    PROJECT_DIR / "data" / "intminet" / "structured_search_meta" / "structured-search_mobility.csv",
    PROJECT_DIR / "data" / "intminet" / "structured_search_meta" / "structured-search-preference.csv"
]

JSON_PATHS=[
    PROJECT_DIR / "data" / "intminet" / "structured_search_meta" / "structured-search_mobility.json",
    PROJECT_DIR / "data" / "intminet" / "structured_search_meta" / "structured-search-preference.json"
]

CSV_PATH = PROJECT_DIR / "data" / "intminet" / "training_data" / "train-data.csv"

MODEL_PATH = PROJECT_DIR  / "models" / "gte-large_abstracts"

PREDICTIONS_OUT = PROJECT_DIR / "output" / "intminet" / "articles_predicted.csv"

ZOTERO_JSON_OUT = PROJECT_DIR / "output" / "intminet" / 'items_import.json'

# Main function
def main():

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

    # Predict

    abstracts_predict_df=add_setfit_predictions(
        abstracts_df,
        model,
        threshold=0.8
    )

    # Store as csv
    PREDICTIONS_OUT.mkdir(parents=True, exist_ok=True)
    abstracts_predict_df.to_csv(PREDICTIONS_OUT, index=False)

    # Select articles and store as Zotero-compatible JSON for import
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

    with open(ZOTERO_JSON_OUT, 'w') as f:
        json.dump(record_out, f)

