import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in tqdm(dataloader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return loss_sum/total, correct/total

def validate(model, dataloader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return correct/total
