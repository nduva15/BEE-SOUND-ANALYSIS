"""
BEESOUND TRAINING - Neural Architecture Trainer
Streams 422,044 samples from Kaggle disk to GPU.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os

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
        
        try:
            # Load only the first 2 seconds to save time/RAM
            y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
            
            # Ensure fixed length (pad if too short)
            if len(y) < self.n_samples:
                y = np.pad(y, (0, self.n_samples - len(y)))
            else:
                y = y[:self.n_samples]
                
            # Convert to Mel Spectrogram (Standard input for Bee architectures)
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize to 0-1
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
            
            return torch.tensor(mel_norm).unsqueeze(0), 0 # Returning 0 as dummy label for now
        except Exception as e:
            # Return a blank sample if file is corrupted
            return torch.zeros((1, 128, 87)), 0

def train_architecture():
    print("🐝 INITIALIZING TRAINING ON 422,044 SAMPLES...")
    
    # 1. Setup Data
    dataset = BeeDataset('train_manifest.csv')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # 2. Setup Device (Always use GPU on Kaggle!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using Device: {device}")

    # 3. Simple Architecture Placeholder (Replace with your ViT or CNN)
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 63 * 42, 2) # Binary: Healthy vs Alert
    ).to(device)

    # 4. Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"📉 Batch {batch_idx} | Loss: {loss.item():.4f}")
            
        if batch_idx > 100: # Limit for first test run
            break

    print("\n✅ Training Test Complete!")

if __name__ == "__main__":
    train_architecture()
