import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from sklearn.metrics import f1_score

import torchvision.transforms as T
import torchvision.models as models


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# HuggingFace Dataset -> PyTorch Dataset
# =========================================================
class HFDatasetAsTorch(Dataset):
    """
    Wrap HuggingFace dataset splits so they work with PyTorch DataLoader.
    Expected columns: 'image' (PIL.Image), 'label' (int).
    """
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]    # PIL image
        label = int(item["label"])

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# =========================================================
# Evaluation metrics
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, macro_f1


# =========================================================
# Training loop (with AMP)
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * y.size(0)

    return running_loss / len(loader.dataset)


# =========================================================
# Model helpers
# =========================================================
def build_resnet18(num_classes: int):
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable(model, train_fc_only: bool):
    for p in model.parameters():
        p.requires_grad = True

    if train_fc_only:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False


# =========================================================
# Main
# =========================================================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------
    # Load dataset
    # -------------------------
    print("Loading dataset timm/resisc45 ...")
    dataset = load_dataset("timm/resisc45")

    train_split = dataset["train"]
    val_split   = dataset["validation"]
    test_split  = dataset["test"]

    print(
        "Dataset sizes:",
        len(train_split),
        len(val_split),
        len(test_split),
    )

    # -------------------------
    # Transforms
    # -------------------------
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = HFDatasetAsTorch(train_split, transform=train_tf)
    val_ds   = HFDatasetAsTorch(val_split, transform=eval_tf)
    test_ds  = HFDatasetAsTorch(test_split, transform=eval_tf)

    # -------------------------
    # DataLoaders
    # -------------------------
    batch_size = 16          # safe for RTX 3050 Ti
    num_workers = 2
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # -------------------------
    # Model
    # -------------------------
    num_classes = 45
    model = build_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    # -------------------------
    # Stage 1: train classifier head
    # -------------------------
    print("\nStage 1: Training classifier head only")
    set_trainable(model, train_fc_only=True)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    epochs_stage1 = 5
    for epoch in range(1, epochs_stage1 + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(
            f"[Stage1 {epoch}/{epochs_stage1}] "
            f"loss={loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    # -------------------------
    # Stage 2: fine-tuning
    # -------------------------
    print("\nStage 2: Fine-tuning entire network")
    set_trainable(model, train_fc_only=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    epochs_stage2 = 5
    for epoch in range(1, epochs_stage2 + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(
            f"[Stage2 {epoch}/{epochs_stage2}] "
            f"loss={loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    # -------------------------
    # Final test
    # -------------------------
    test_acc, test_f1 = evaluate(model, test_loader, device)

    print("\n=== Final Test Results ===")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}")

    # -------------------------
    # Save model
    # -------------------------
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/resnet18_best.pth")
    print("Saved model to results/resnet18_best.pth")


if __name__ == "__main__":
    main()
