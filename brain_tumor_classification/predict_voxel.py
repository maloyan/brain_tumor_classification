import sys
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brain_tumor_classification.dataset import *
from brain_tumor_classification.model import BrainTumorClassificationModel

def predict(model_path, df, mri_types, split, config):

    data_retriever = VoxelBrainDataset(
        patients=df.index.values,
        targets=None,
        mri_types=mri_types, 
        split=split,
        data_directory=config["data_directory"],
    )

    data_loader = DataLoader(
        data_retriever,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    #model = BrainTumorClassificationModel()
    model = torch.load(model_path)
    model.to(config["device"])

    #checkpoint = torch.load(modelfile)
    #model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_pred = []
    ids = []

    for e, batch in enumerate(data_loader, 1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = (
                torch.sigmoid(model(batch["X"].to(config["device"])))
                .cpu()
                .numpy()
                .squeeze()
            )
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
            ids.extend(batch["id"].numpy().tolist())

    preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
    preddf = preddf.set_index("BraTS21ID")
    return preddf

with open(sys.argv[1], "r") as f:
    config = json.load(f)

submission = pd.read_csv(
    f"{config['csv_data_directory']}/sample_submission.csv", index_col="BraTS21ID"
)

submission["MGMT_value"] = 0

pred = predict(
    f"checkpoints/voxel_{config['model_name']}.pt", 
    submission, 
    config["mri_types"], 
    split="test", 
    config=config
)
submission["MGMT_value"] = pred["MGMT_value"]

submission["MGMT_value"].to_csv("submission.csv")
