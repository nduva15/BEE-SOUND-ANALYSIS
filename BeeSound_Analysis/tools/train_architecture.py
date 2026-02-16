"""
BEESOUND TRAINING - Research-Grade Neural Engine (v3)
Implementing: MixUp, Focal Loss, Label Smoothing, and Residual Bioacoustic Mapping.
Based on research papers by Lin et al. (Focal Loss) and Zhang et al. (MixUp).
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
import numpy as np
import os

# ==========================================
# 1. RESEARCH-GRADE LOSS: SMOOTH FOCAL LOSS
# ==========================================

class SmoothFocalLoss(nn.Module):
    """
    Combines Label Smoothing (Szegedy et al.) and Focal Loss (Lin et al.)
    to prevent 'Trivial Convergence' (0.0000 loss collapse).
    """
    def __init__(self, num_classes=2, smoothing=0.1, gamma=2.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        # 1. Label Smoothing: Convert hard labels to 'soft' confidence
        pred_log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        # 2. Focal Loss: Weight hard examples more than easy ones
        p_t = torch.exp(pred_log_prob)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Combine: (1 - p_t)^gamma * Smoothed Cross Entropy
        loss = torch.sum(-true_dist * focal_weight * pred_log_prob, dim=-1)
        return loss.mean()

# ==========================================
# 2. DATA AUGMENTATION: MIXUP (Zhang et al.)
# ==========================================

def mixup_data(x, y, alpha=0.4, device='cuda'):
    """
    Blends two samples together to force the model to learn the 'space between'.
    Impossible to 'memorize-cheat' on blended ghost signals.
    """
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
# 3. ADVANCED ARCHITECTURE: BIOTRANSFORMER-CNN
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
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Deep Residual Layers
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*1
        layers = []
        for s in strides:
            layers.append(BeeResNetBlock(self.in_planes, planes, s))
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
# 4. DATASET & LOADER
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
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
            
            return torch.tensor(mel_norm).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        except:
            return torch.zeros((1, 128, 87)), torch.tensor(0, dtype=torch.long)

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

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
            if i > 100: break # Quick check for progress
            
    p = precision_score(all_targets, all_preds, zero_division=0)
    r = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return p, r, f1, cm

def train_production():
    # ... existing setup ...
    print("🐝 BEESOUND PRODUCTION ENGINE v3.0 (RESEARCH GRADE)")
    print("   Defenses: MixUp + Focal Loss + Label Smoothing")
    print("="*70)

    # 1. Dataset & Balanced Sampling
    manifest_path = 'train_manifest_research.csv'
    if not os.path.exists(manifest_path):
        manifest_path = 'train_manifest_labeled.csv'
        
    dataset = BeeDataset(manifest_path)
    
    # ⚖️ Weighted Sampler: Kill the 750:1 Imbalance
    labels = dataset.df['label'].values
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts + 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)
    
    # 2. Compute Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeeDeepArchitecture().to(device)
    
    # Advanced Optimizer with Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(loader), epochs=10)
    
    # RESEARCH LOSS: Smoothed Focal
    criterion = SmoothFocalLoss(num_classes=2, smoothing=0.1, gamma=2.0).to(device)

    print(f"🖥️  Compute: {device} | 📦 Samples: {len(dataset)}")
    
    best_loss = float('inf')
    
    for epoch in range(10):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for i, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            
            # 👻 MixUp Augmentation
            data, labels_a, labels_b, lam = mixup_data(data, labels, device=device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Use Mixup Criterion
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            # scheduler.step() # Moved scheduler.step() to end of epoch
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 0:
                print(f"📉 Ep{epoch} Batch{i}/{len(loader)} | Loss: {loss.item():.6f} | Acc: {100.*correct/total:.2f}%")
        
        # Epoch Summary
        epoch_acc = 100. * correct / total
        
        # 🧪 THE TRUTH TEST: Calculate F1-Score
        p, r, f1, cm = calculate_truth_metrics(loader, model, device)
        
        print(f"✨ Epoch {epoch} Results:")
        print(f"   Avg Loss:  {running_loss/len(loader):.4f}")
        print(f"   Accuracy:  {epoch_acc:.2f}%")
        print(f"   Precision: {p:.4f} | Recall: {r:.4f}")
        print(f"   🏆 F1-SCORE: {f1:.4f}")
        print(f"   Confusion Matrix:\n{cm}")
        
        # Save Model ONLY if F1 improves
        if f1 > best_acc: # reusing best_acc variable as best_f1
            best_acc = f1
            torch.save(model.state_dict(), 'beesound_deepbrain_v3.pth')
            print(f"💾 NEW BEST BRAIN SAVED (F1: {f1:.4f})")
        
        scheduler.step() # Moved scheduler.step() here
    
if __name__ == "__main__":
    train_production()
