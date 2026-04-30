import os
import glob
import numpy as np
import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torchvision.transforms as T
import torchvision.transforms.functional as F

from models import DualChannelDeepfakeDetector


import glob

def load_processed_data(data_dir):
    # get all of the data from google drive to aviod rerunning the extract frames again and again.
    data_pattern = os.path.join(data_dir, "*_data_b*.npy")
    label_pattern = os.path.join(data_dir, "*_label_b*.npy")
    
    data_files = sorted(glob.glob(data_pattern))
    label_files = sorted(glob.glob(label_pattern))
    X = np.concatenate([np.load(f) for f in data_files])
    y = np.concatenate([np.load(f) for f in label_files])
    print(f"Data loading complete!")
    print(f"Total Features (X) shape: {X.shape}")
    print(f"Total Labels (y) shape: {y.shape}")
    return X, y



## Find Anchor， positive, negative within patch, and calculate their trip loss


def get_triplet_loss(embeddings, labels, criterion_tri):
    ##For each anchor, this function selects:
    ## - the closest positive sample
    ## - a semi-hard negative sample, i.e. a negative that is farther than
    ##  the positive but still as close as possible to the anchor
    ##
    anchors, positives, negatives = [], [], []
    batch_size = labels.size(0)

    # calculate the distance between embedding 
    dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

    for i in range(batch_size):
        # Identify the positive and negatice sample
        pos_mask = (labels == labels[i])
        neg_mask = (labels != labels[i])
        pos_mask[i] = False  

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # select the cloest positive sample
        pos_idx = dist[i][pos_mask].argmin()
        pos_emb = embeddings[pos_mask][pos_idx]

        # Semi-hard negative：farther than the positive, but still as close as possible）
        d_pos = dist[i][pos_mask][pos_idx].item()
        hard_neg_mask = neg_mask & (dist[i] > d_pos)

        if hard_neg_mask.sum() > 0:
            neg_idx = dist[i][hard_neg_mask].argmin()
            neg_emb = embeddings[hard_neg_mask][neg_idx]
        else:
            #  pick a negative 
            neg_idx = dist[i][neg_mask].argmin()
            neg_emb = embeddings[neg_mask][neg_idx]

        anchors.append(embeddings[i])
        positives.append(pos_emb)
        negatives.append(neg_emb)

    if len(anchors) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return criterion_tri(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

class DeepfakeVideoDataset(Dataset):
    def __init__(self, X, y, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), train=True):
      self.X = X
      self.y = y
      self.mean = torch.tensor(mean).view(1, 3, 1, 1)
      self.std = torch.tensor(std).view(1, 3, 1, 1)
      self.train = train
      self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
      return len(self.X)

    def __getitem__(self, idx):

      video = self.X[idx]   # (T, H, W, C)
      label = self.y[idx]

      # numpy -> torch 
      video = torch.tensor(video, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()

      # randomly do data argumentaion for training dataset, while training for reduce overfiting.

      if self.train:
          # greyscale the image with 0.2 probability
          if random.random() > 0.8:
              video = F.rgb_to_grayscale(video, num_output_channels=3)

          # randomly rotatio to trainig data
          if random.random() > 0.5:
              angle = random.uniform(-10, 10)
              video = torch.stack([F.rotate(video[i], angle) for i in range(video.shape[0])])

          # applying Gaussian Noise
          if random.random() > 0.5:
              noise = torch.randn_like(video) * 0.02
              video = video + noise

          if random.random() > 0.5:
              video = F.hflip(video)

          if random.random() > 0.5:
              params = T.ColorJitter.get_params(
                  self.color_jitter.brightness,
                  self.color_jitter.contrast,
                  self.color_jitter.saturation,
                  self.color_jitter.hue
              )
              _, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
          
              frames = []
              for i in range(video.shape[0]):
                  f = video[i].clone()
                  f = F.adjust_brightness(f, brightness_factor)
                  f = F.adjust_contrast(f, contrast_factor)
                  f = F.adjust_saturation(f, saturation_factor)
                  f = F.adjust_hue(f, hue_factor)
                  frames.append(f)
              video = torch.stack(frames)

      # normalization
      video = (video - self.mean) / self.std
      return video, torch.tensor(label, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, 
                    triplet_weight,device,triplet_criterion=None,):  
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for videos, labels in loader:
        videos = videos.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits, embeddings, _ = model(videos)

        loss_bce = criterion(logits, labels)

        # if statement use to do comparesion between these two loss, to see how they affect the model
        if triplet_criterion is not None:
            loss_tri = get_triplet_loss(embeddings, labels, triplet_criterion)
            loss = loss_bce + triplet_weight * loss_tri
        else:
            loss = loss_bce

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * videos.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return epoch_loss, acc, f1


def evaluate(model, loader, criterion,device):
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device).float()  
            labels = labels.to(device).float()
            logits, _, _ = model(videos)

            loss = criterion(logits, labels)
            total_loss += loss.item() * videos.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # calculate the performance metric
    epoch_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return epoch_loss, acc, f1, all_labels, all_probs


def main():
    # change this path to your processed data path, which is the output of the extract_frames.py, and make sure you have run the extract_frames.py to get the processed data.
    save_dir = "./data/"
    os.makedirs("results", exist_ok=True)

    X, y = load_processed_data(save_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_dataset = DeepfakeVideoDataset(X_train, y_train, train=True)
    val_dataset = DeepfakeVideoDataset(X_val, y_val, train=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

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


    criterion = nn.BCEWithLogitsLoss()
    criterion_triplet = nn.TripletMarginLoss(margin=0.3)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=4e-4,
        weight_decay=1e-2
    )

    num_epochs = 10
    best_val_f1 = 0.0
    patience = 3
    no_improve = 0
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion,device=device,
            triplet_criterion=criterion_triplet,
            triplet_weight=0.3
        )

        val_loss, val_acc, val_f1, labels, probs = evaluate(
            model, val_loader, criterion,device=device
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), "results/best_model.pth")
            print("Saved best model.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()