import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, criterion, device, scheduler):
    sum_loss = 0
    model.train()

    for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        X = batch["X"].to(device)
        targets = batch["y"].to(device)

        optimizer.zero_grad()
        outputs = model(X).squeeze(1)

        loss = criterion(outputs, targets)
        loss.backward()
        sum_loss += loss.detach().item()

        optimizer.step()

    return sum_loss / len(data_loader)


def eval_fn(data_loader, model, criterion, device):
    model.eval()
    sum_loss = 0
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            X = batch["X"].to(device)
            targets = batch["y"].to(device)

            outputs = model(X).squeeze(1)
            loss = criterion(outputs, targets)
            sum_loss += loss.detach().item()

            fin_targets.extend(targets.tolist())
            fin_outputs.extend(outputs.tolist())

    fin_targets = [1 if x > 0.5 else 0 for x in fin_targets]
    auc = roc_auc_score(fin_targets, fin_outputs)
    return sum_loss / len(data_loader), auc
