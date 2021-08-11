import json
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split

from brain_tumor_classification.dataset import BrainTumorClassificationDataset
from brain_tumor_classification.engine import eval_fn, train_fn
from brain_tumor_classification.model import BrainTumorClassificationModel
from brain_tumor_classification.predict import predict
from brain_tumor_classification.utils import set_seed

with open(sys.argv[1], "r") as f:
    config = json.load(f)

wandb.init(config=config, project="feta")

set_seed(config["seed"])

train_df = pd.read_csv(f"{config['data_directory']}/train_labels.csv")

df_train, df_valid = train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)

if config["mri_type"] == "all":
    train_list = []
    valid_list = []
    for mri_type in config["mri_types"]:
        df_train.loc[:, "MRI_Type"] = mri_type
        train_list.append(df_train.copy())
        df_valid.loc[:, "MRI_Type"] = mri_type
        valid_list.append(df_valid.copy())

    df_train = pd.concat(train_list)
    df_valid = pd.concat(valid_list)
else:
    df_train.loc[:, "MRI_Type"] = config["mri_type"]
    df_valid.loc[:, "MRI_Type"] = config["mri_type"]

train_data = BrainTumorClassificationDataset(
    df_train["BraTS21ID"].values,
    df_train["MGMT_value"].values,
    df_train["MRI_Type"].values,
    augment=True,
)

valid_data = BrainTumorClassificationDataset(
    df_valid["BraTS21ID"].values,
    df_valid["MGMT_value"].values,
    df_valid["MRI_Type"].values,
)

train_loader = torch.utils.DataLoader(
    train_data,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
)

valid_loader = torch.utils.DataLoader(
    valid_data,
    batch_size=4,
    shuffle=False,
    num_workers=8,
)

model = BrainTumorClassificationModel()
model = torch.nn.DataParallel(model, device_ids=config["device_ids"])

criterion = F.binary_cross_entropy_with_logits

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    factor=config["reduce_factor"], patience=config["patience"]
)

best_loss = 1000
for _ in range(config["epochs"]):
    train_loss = train_fn(
        train_loader, model, optimizer, criterion, config["device"], scheduler
    )
    val_loss, val_roc_auc = eval_fn(valid_loader, model, criterion, config["device"])
    if val_loss < best_loss:
        torch.save(
            model.module, f"checkpoints/{config['mri_type']}_{config['model_name']}.pt"
        )
    wandb.log(
        {"train_loss": train_loss, "val_loss": val_loss, "val_roc_auc": val_roc_auc}
    )

submission = pd.read_csv(
    f"{config['data_directory']}/sample_submission.csv", index_col="BraTS21ID"
)

submission["MGMT_value"] = 0
for mtype in config["mri_types"]:
    pred = predict(model, submission, mtype, split="test")
    submission["MGMT_value"] += pred["MGMT_value"]

submission["MGMT_value"] /= len(modelfiles)
submission["MGMT_value"].to_csv("submission.csv")
