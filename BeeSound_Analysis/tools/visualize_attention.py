import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys

# Add parent dir for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.train_architecture import BeeDeepArchitecture, BeeDataset

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class=None):
        input_image = input_image.requires_grad_(True)
        logit = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(logit, dim=1).item()
        
        self.model.zero_grad()
        logit[0, target_class].backward()
        
        # Average gradients across spatial dimensions (GAP)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Apply ReLU to keep only positive influence
        
        # Upsample to match input size
        cam = F.interpolate(cam, size=(input_image.shape[2], input_image.shape[3]), mode='bilinear', align_corners=False)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.detach().cpu().numpy()[0, 0], logit

def visualize_bee_attention(weights_path, audio_path, output_path="docs/gradcam_analysis.png"):
    """
    Generates a Grad-CAM heatmap over the mel-spectrogram.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BeeDeepArchitecture().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Determine target layer (the last residual layer)
    target_layer = model.layer3[-1].bn2
    cam_gen = GradCAM(model, target_layer)

    # 1. Process Sample
    sr = 22050
    duration = 2.0
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
    input_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)

    # 2. Generate CAM
    cam, logit = cam_gen.generate(input_tensor)
    pred_class = torch.argmax(logit, dim=1).item()
    confidence = F.softmax(logit, dim=1)[0, pred_class].item()

    # 3. Plotting
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw Spectrogram
    img1 = librosa.display.specshow(mel_db * 40 - 40, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax1)
    ax1.set_title(f"Acoustic Input: {os.path.basename(audio_path)}")
    plt.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # Grad-CAM Overlay
    ax2.imshow(mel_db, origin='lower', aspect='auto', cmap='gray')
    ax2.imshow(cam, origin='lower', aspect='auto', cmap='jet', alpha=0.5)
    
    label_map = {0: "Healthy/Dormant", 1: "ALERT: Queenless/Swarm"}
    ax2.set_title(f"AI Focus Map (Grad-CAM) | Prediction: {label_map[pred_class]} ({confidence:.2%})")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Mel Frequency Bins (0-8kHz)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Grad-CAM Analysis exported to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/visualize_attention.py <weights.pth> <audio.wav>")
    else:
        visualize_bee_attention(sys.argv[1], sys.argv[2])
