import torch
import numpy as np
import tensorflow as tf
import sys
import os

# Add parent dir for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.train_architecture import BeeDeepArchitecture

def verify_parity(pytorch_path, tflite_path):
    print("⚖️  Starting Model Parity Verification...")
    
    # 1. Load PyTorch Model
    device = torch.device('cpu')
    pt_model = BeeDeepArchitecture()
    pt_model.load_state_dict(torch.load(pytorch_path, map_location=device))
    pt_model.eval()
    
    # 2. Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 3. Generate Random Test Probe
    # Shape: (1, 1, 128, 87)
    test_input = np.random.randn(1, 1, 128, 87).astype(np.float32)
    
    # Run PyTorch Inference
    with torch.no_grad():
        pt_output = pt_model(torch.from_numpy(test_input)).numpy()
    
    # Run TFLite Inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # 4. Compare Results
    # Since Int8 quantization involves rounding, we expect some difference
    # but the argmax (class prediction) should ideally match.
    pt_class = np.argmax(pt_output)
    tf_class = np.argmax(tflite_output)
    
    diff = np.abs(pt_output - tflite_output).mean()
    
    print(f"📊 PyTorch Output: {pt_output}")
    print(f"📊 TFLite Output:  {tflite_output}")
    print(f"📉 Mean Absolute Error: {diff:.6f}")
    
    if pt_class == tf_class:
        print("✅ SUCCESS: Class Prediction Parity Maintained.")
    else:
        print("⚠️ WARNING: Class Prediction Mismatch. Check quantization calibration.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/verify_parity.py <model.pth> <model.tflite>")
    else:
        verify_parity(sys.argv[1], sys.argv[2])
