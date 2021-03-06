import sys
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brain_tumor_classification.dataset import BrainTumorClassificationDataset
from brain_tumor_classification.model import BrainTumorClassificationModel

def predict(modelfile, df, mri_type, split, config):
    df.loc[:, "MRI_Type"] = mri_type

    data_retriever = BrainTumorClassificationDataset(
        df.index.values, 
        mri_type=df["MRI_Type"].values, 
        split=split,
        data_directory=config["data_directory"],
    )

    data_loader = DataLoader(
        data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=8,
    )

    #model = BrainTumorClassificationModel()
    model = torch.load(modelfile)
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
    f"{config['data_directory']}/sample_submission.csv", index_col="BraTS21ID"
)

submission["MGMT_value"] = 0
for mtype in config["mri_types"]:
    pred = predict(f"checkpoints/{config['mri_type']}_{config['model_name']}.pt", submission, mtype, split="test", config=config)
    submission["MGMT_value"] += pred["MGMT_value"]

submission["MGMT_value"] /= len(config["mri_types"])
submission["MGMT_value"].to_csv("submission.csv")
