import json
import sys

import pandas as pd

from brain_tumor_classification.utils import *

with open(sys.argv[1], "r") as f:
    config = json.load(f)

set_seed(config["seed"])

train_df = pd.read_csv(f"{config["data_directory"]}/train_labels.csv")
display(train_df)

df_train, df_valid = sk_model_selection.train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=12, 
    stratify=train_df["MGMT_value"],
)
