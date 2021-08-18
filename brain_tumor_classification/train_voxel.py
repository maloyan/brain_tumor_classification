import json
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from brain_tumor_classification.dataset import *
from brain_tumor_classification.engine import eval_fn, train_fn
from brain_tumor_classification.model import BrainTumorClassificationModel
from brain_tumor_classification.utils import set_seed

with open(sys.argv[1], "r") as f:
    config = json.load(f)

wandb.init(
    config=config, 
    project=config["project"],
    name=f"voxel_{config['model_name']}"
)

set_seed(config["seed"])

train_df = pd.read_csv(f"{config['data_directory']}/train_labels.csv")

df_train, df_valid = train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)

train_data = VoxelBrainDataset(
    patients=df_train["BraTS21ID"].values,
    targets=df_train["MGMT_value"].values,
    mri_types=config["mri_types"],
    data_directory=config["data_directory"]
)

valid_data = VoxelBrainDataset(
    patients=df_valid["BraTS21ID"].values,
    targets=df_valid["MGMT_value"].values,
    mri_types=config["mri_types"],
    data_directory=config["data_directory"]
)

train_loader = DataLoader(
    train_data,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
)

valid_loader = DataLoader(
    valid_data,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
)

model = BrainTumorClassificationModel(config["backbone"])
model.to(config["device"])
model = torch.nn.DataParallel(model, device_ids=config["device_ids"])

criterion = F.binary_cross_entropy_with_logits

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=config["reduce_factor"], patience=config["patience"], verbose=True
)

best_loss = 1000
for _ in range(config["epochs"]):
    train_loss = train_fn(
        train_loader, model, optimizer, criterion, config["device"], scheduler
    )
    val_loss, val_roc_auc = eval_fn(
        valid_loader, model, criterion, config["device"]
    )
    
    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(
            model.module, f"checkpoints/voxel_{config['model_name']}.pt"
        )
    wandb.log(
        {"train_loss": train_loss, "val_loss": val_loss, "val_roc_auc": val_roc_auc}
    )
