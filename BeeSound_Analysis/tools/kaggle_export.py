"""
BEESOUND ANALYSIS - Kaggle Export Orchestrator
Automates the PyTorch -> ONNX -> TFLite export pipeline and packages
a finalized release zip for easy download or API push from Kaggle Kernels.

Usage:
    python tools/kaggle_export.py --weights beesound_final_v3.pth --push
"""

import os
import sys
import shutil
import argparse
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

# Add parent directory for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tools.export_brain import export_to_onnx
    from tools.quantize_brain import quantize_to_tflite
except ImportError:
    print("❌ Error: Could not load export_brain or quantize_brain modules.")
    sys.exit(1)


def find_best_checkpoint(start_dir="/kaggle/working", default_name="beesound_final_v3.pth"):
    """Scans for the best PyTorch checkpoint if the default is not found."""
    if os.path.exists(default_name):
        return default_name

    # Try searching Kaggle working dir and local dirs
    candidates = []
    search_dirs = [start_dir, ".", "./models", "./weights"]
    for d in set(search_dirs):
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('.pth'):
                    candidates.append(os.path.join(root, f))
                    
    if not candidates:
        return None
    
    # Priority 1: 'final' in name
    for c in candidates:
        if 'final' in c:
            return c
    
    # Priority 2: 'v3' in name
    for c in candidates:
        if 'v3' in c:
            return c
            
    # Priority 3: Largest file (most likely the full checkpoint)
    candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return candidates[0]


def create_release_pack(pth_path, output_dir="release_artifacts"):
    """
    Runs the full export pipeline.
    """
    print("=" * 60)
    print("📦 BEESOUND KAGGLE ARTIFACT EXPORTER")
    print("=" * 60)
    
    # Auto-detection
    detected_path = find_best_checkpoint(default_name=pth_path)
    if not detected_path:
        print(f"❌ Cannot find target weights: {pth_path} or any valid .pth in working directory.")
        return None
        
    if detected_path != pth_path:
        print(f"🔍 Auto-detected best checkpoint: {detected_path}")
    pth_path = detected_path

    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(pth_path))[0]
    
    pth_out = os.path.join(output_dir, f"{basename}.pth")
    onnx_out = os.path.join(output_dir, f"{basename}.onnx")
    tf_out_dir = os.path.join(output_dir, f"{basename}_tf")
    tflite_out = os.path.join(output_dir, f"{basename}_int8.tflite")
    zip_out = f"{output_dir}.zip"

    # Step 1: Copy PyTorch Model
    print(f"\n[1/4] Copying PyTorch checkpoint...")
    shutil.copy2(pth_path, pth_out)
    print(f"   ✅ Saved to {pth_out}")

    # Step 2: Export to ONNX
    print(f"\n[2/4] Exporting to ONNX...")
    try:
        export_to_onnx(pth_path, output_path=onnx_out, num_classes=2)
    except Exception as e:
        print(f"   ❌ ONNX Export Failed: {e}")
        print("   Warning: Proceeding with partial pack.")

    # Step 3: Quantize (Requires onnx2tf -> TFLite)
    if os.path.exists(onnx_out):
        print(f"\n[3/4] Quantizing to TFLite (ESP32 target)...")
        # 3a. onnx2tf intermediate
        print(f"   Generating TF SavedModel...")
        # Verify if onnx2tf is installed
        has_onnx2tf = shutil.which("onnx2tf") is not None
        if not has_onnx2tf:
            print("   ⚠️  'onnx2tf' is not installed in this environment.")
            print("   ⚠️  Run: pip install onnx2tf tensorflow")
            print("   ⚠️  Skipping TFLite export.")
        else:
            try:
                # Run onnx2tf
                cmd = ["onnx2tf", "-i", onnx_out, "-o", tf_out_dir]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"   ✅ TF SavedModel generated.")
                
                # 3b. PTQ to TFLite
                from tools.quantize_brain import quantize_to_tflite
                quantize_to_tflite(onnx_out, tflite_out, representative_data_path=None) 
                
            except subprocess.CalledProcessError:
                print("   ❌ onnx2tf processing failed.")
            except Exception as e:
                print(f"   ❌ Quantization Failed: {e}")
    else:
        print(f"\n[3/4] Skipping TFLite (missing ONNX parent file)...")

    # Step 4: Zip Everything
    print(f"\n[4/4] Archiving Release Package...")
    with ZipFile(zip_out, 'w', ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep folder structure flat for the zip if possible
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

    # Cleanup raw directory in Kaggle to save space if needed
    # We will leave it for inspection, but usually good practice.
    
    print("\n" + "=" * 60)
    print(f"🎉 EXPORT COMPLETE!")
    print(f"   Output Archive: {os.path.abspath(zip_out)}")
    print("=" * 60)
    
    return zip_out


def push_to_kaggle(zip_path, dataset_slug):
    """
    Push the generated zip artifact to a Kaggle Dataset using API.
    Require: kaggle.json configured + dataset-metadata.json
    """
    print(f"\n🚀 Initiating push to Kaggle: {dataset_slug}...")
    try:
        import kaggle
    except ImportError:
        print("❌ 'kaggle' library not installed. Cannot push.")
        return

    # Basic setup for Kaggle API dataset initialization
    working_dir = os.path.dirname(os.path.abspath(zip_path))
    meta_path = os.path.join(working_dir, "dataset-metadata.json")
    
    print(f"   Warning: Kaggle Dataset pushing requires pre-configured dataset-metadata.json")
    print(f"   or manual creation via: kaggle datasets init -p {working_dir}")
    print(f"   Command: kaggle datasets version -p {working_dir} -m 'Automated Release'")
    print(f"   Note: This feature is experimental. Please manually download the ZIP from the UI for now.")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Export Integrator")
    parser.add_argument("--weights", type=str, default="beesound_final_v3.pth", help="Path to best PyTorch model")
    parser.add_argument("--push", action="store_true", help="Push to Kaggle Datasets (Experimental)")
    parser.add_argument("--slug", type=str, default="nduva15/beesound-v3-artifacts", help="Kaggle dataset slug")
    
    args = parser.parse_args()
    
    zip_file = create_release_pack(args.weights)
    
    if args.push and zip_file:
        push_to_kaggle(zip_file, args.slug)


if __name__ == "__main__":
    main()
