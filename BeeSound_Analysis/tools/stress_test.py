import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os

def add_noise(audio_tensor, noise_level):
    """
    Injects random Gaussian noise into the mel-spectrogram tensor.
    noise_level: 0.0 (Clean) to 1.0 (Extreme Static)
    """
    if noise_level <= 0:
        return audio_tensor
    noise = torch.randn_like(audio_tensor)
    return audio_tensor + (noise * noise_level)

def run_stress_test(model, loader, device='cuda'):
    """
    Evaluates model stability across increasing noise thresholds.
    This is critical for proving field-readiness in windy/rainy conditions.
    """
    print("\n🌪️  ORCHESTRATING SCIENTIFIC STRESS TEST...")
    
    # Noise levels from 0% to 50% intensity
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    f1_scores = []
    
    model.eval()
    
    for level in noise_levels:
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 💥 Inject artificial environmental interference
                noisy_inputs = add_noise(inputs, level)
                
                outputs = model(noisy_inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                if i > 50: break # Standardized probe size for speed
        
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        f1_scores.append(f1)
        print(f"   [Intensity {level*100:2.0f}%] -> F1-Score: {f1:.4f}")
        
    return noise_levels, f1_scores

def plot_stress_results(noise_levels, f1_scores, output_path="docs/stress_test_curve.png"):
    """
    Generates a scientific plot of model degradation vs nose.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(noise_levels)*100, f1_scores, marker='o', linestyle='-', color='#FFD700', linewidth=2)
    plt.fill_between(np.array(noise_levels)*100, f1_scores, color='#FFD700', alpha=0.2)
    
    plt.title("BeeSound v3.1: Acoustic Robustness Analysis", fontsize=14)
    plt.xlabel("Injected Noise Intensity (%)", fontsize=12)
    plt.ylabel("F1-Score (Vitality Detection)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.1)
    
    plt.savefig(output_path)
    print(f"📈 Robustness Curve exported to {output_path}")

if __name__ == "__main__":
    print("⚠️ This tool is intended to be called from the Training Engine after Epoch 1.")
