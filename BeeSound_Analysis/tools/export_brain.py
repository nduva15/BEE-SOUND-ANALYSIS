import torch
import torch.onnx
import sys
import os

# Append current directory to path so we can import models from train_architecture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.train_architecture import BeeDeepArchitecture

def export_to_onnx(model_path, num_classes=2):
    print(f"📦 Loading Best Brain from: {model_path}...")
    
    # 1. Initialize the Model Architecture
    model = BeeDeepArchitecture(num_classes=num_classes)
    
    # 2. Load the trained weights
    # Note: Use map_location='cpu' to ensure it works even if saved on GPU
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    model.eval() # Set to evaluation mode (Crucial!)
    
    # 3. Create a Dummy Input
    # Shape based on BeeDataset: (Batch, Channels, Mel_Bins, Time_Steps)
    # The dataset generates (1, 128, 87)
    dummy_input = torch.randn(1, 1, 128, 87) 
    
    # 4. Export to ONNX
    output_path = "models/bee_brain_v3.onnx"
    os.makedirs("models", exist_ok=True)
    
    print(f"🚀 Exporting to ONNX format...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12, # Higher opset for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ SUCCESS: Brain exported to {output_path}")
    print("   Next Step: Quantize to .tflite for ESP32 Deployment.")

if __name__ == "__main__":
    # Check if a specific path was provided, else use default
    target_weights = "beesound_final_v3.pth" 
    if len(sys.argv) > 1:
        target_weights = sys.argv[1]
        
    if os.path.exists(target_weights):
        export_to_onnx(target_weights)
    else:
        print(f"⚠️ Looking for weights at: {target_weights}")
        print("   Please provide the path to your .pth file.")
