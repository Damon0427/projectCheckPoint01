import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models import DualChannelDeepfakeDetector
from training import load_processed_data, DeepfakeVideoDataset


def evaluate_model(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            logits, _, _ = model(videos)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_labels, all_preds, all_probs


def save_metrics(y_true, y_pred, y_prob, save_path):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    with open(save_path, "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n")

    print(f"Saved metrics to {save_path}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved ROC curve to {save_path}")


def plot_confusion_matrix_figure(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


def main():
    os.makedirs("results", exist_ok=True)

    data_dir = "./data/"
    model_path = "results/best_model.pth"

    X, y = load_processed_data(data_dir)

    # 和 training.py 保持一致
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    val_dataset = DeepfakeVideoDataset(X_val, y_val, train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # get the apple MPS device
    if torch.backends.mps.is_available():
                device = torch.device("mps")
                print("using Mac MPS ")
    else:
        device = torch.device("cpu")
        print("using CPU !")

    model = DualChannelDeepfakeDetector(
        model_name="google/vit-base-patch16-224",
        freeze_vit=True,
        dropout=0.3
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)

    save_metrics(y_true, y_pred, y_prob, "results/metrics.txt")
    plot_roc_curve(y_true, y_prob, "results/roc_curve.png")
    plot_confusion_matrix_figure(y_true, y_pred, "results/confusion_matrix.png")


if __name__ == "__main__":
    main()