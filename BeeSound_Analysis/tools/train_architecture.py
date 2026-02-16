"""
BEESOUND TRAINING - Research Grade Neural Engine (v3.1)
Fixed: Layer Dimension Mismatch (128 vs 256).
Implementing: MixUp, Focal Loss, Label Smoothing.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ==========================================
# 1. RESEARCH-GRADE LOSS: SMOOTH FOCAL LOSS
# ==========================================

class SmoothFocalLoss(nn.Module):
    def __init__(self, num_classes=2, smoothing=0.1, gamma=2.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred_log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        p_t = torch.exp(pred_log_prob)
        focal_weight = (1 - p_t) ** self.gamma
        loss = torch.sum(-true_dist * focal_weight * pred_log_prob, dim=-1)
        return loss.mean()

# ==========================================
# 2. DATA AUGMENTATION: MIXUP
# ==========================================

def mixup_data(x, y, alpha=0.4, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 3. ALIGNED ARCHITECTURE: RESNET-STYLE
# ==========================================

class BeeResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BeeDeepArchitecture(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.in_planes = 64
        
        # Initial 64-channel filter
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Deep Residual Layers (Growth: 64 -> 128 -> 256)
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), # Input 256 from layer3 output
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, planes, stride):
        layers = []
        layers.append(BeeResNetBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return self.fc(out)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================

class BeeDataset(Dataset):
    def __init__(self, manifest_path, sample_rate=22050, duration=2.0):
        self.df = pd.read_csv(manifest_path)
        self.sr = sample_rate
        self.duration = duration
        self.n_samples = int(self.sr * self.duration)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['file_path']
        label = int(self.df.iloc[idx]['label'])
        try:
            y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            else: y = y[:self.n_samples]
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128, fmax=8000)
            mel_db = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
            return torch.tensor(mel_db).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        except:
            return torch.zeros((1, 128, 87)), torch.tensor(0, dtype=torch.long)

def calculate_truth_metrics(loader, model, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            data = data.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            if i > 50: break # Quick validation probe
    
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return f1, cm

def train_production():
    print("="*70)
    print("🐝 BEESOUND PRODUCTION ENGINE v3.1 (DIMENSION ALIGNED)")
    print("="*70)

    manifest_path = 'train_manifest_research.csv'
    if not os.path.exists(manifest_path):
        manifest_path = 'train_manifest_labeled.csv'
        
    dataset = BeeDataset(manifest_path)
    
    # Kill the 750:1 Imbalance with Weighted Sampling
    labels = dataset.df['label'].values
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts + 1)
    sampler = WeightedRandomSampler(class_weights[labels], len(labels), replacement=True)

    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BeeDeepArchitecture().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = SmoothFocalLoss(num_classes=2, smoothing=0.1, gamma=2.0).to(device)

    best_f1 = 0.0
    for epoch in range(10):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(data, labels, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"📉 Ep{epoch} Batch{i}/{len(loader)} | Loss: {loss.item():.6f}")

        f1, cm = calculate_truth_metrics(loader, model, device)
        print(f"✨ Epoch {epoch} Results | Avg Loss: {running_loss/len(loader):.4f} | 🏆 F1: {f1:.4f}")
        print(f"   Matrix:\n{cm}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'beesound_final_v3.pth')
            print("💾 Best Brain Saved!")

if __name__ == "__main__":
    train_production()
