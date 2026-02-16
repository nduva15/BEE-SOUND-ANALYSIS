"""
BEESOUND ANALYSIS - Dataset Inspector (HDF5)
Adapted for local use with the BeeTogether Dataset (NUHIVE, TBON, etc.)
"""

import pandas as pd
import os
import argparse

def inspect_datasets(base_path):
    # Standard BeeTogether files
    files = ["NUHIVE.h5", "TBON.h5", "SBCM.h5", "BAD.h5"]
    
    print(f"📂 Checking contents of {base_path}...")
    
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        print(f"⚠️  Path did not exist. Created: {base_path}")
        print("👉 Please drop your .h5 files here!")
        return

    existing_files = os.listdir(base_path)
    print(f"   Found files: {existing_files}")
    
    # Check structure without crashing RAM
    for file_name in files:
        full_path = os.path.join(base_path, file_name)
        
        if not os.path.exists(full_path):
            print(f"\n❌ Missing: {file_name}")
            continue
            
        try:
            # Open in read-only mode to inspect keys
            print(f"\n🔍 Inspecting: {file_name}")
            with pd.HDFStore(full_path, mode='r') as store:
                print(f"   Keys found: {store.keys()}")
                
                # Check for common keys
                key_to_check = None
                if '/bee_audio' in store.keys():
                    key_to_check = 'bee_audio'
                elif len(store.keys()) > 0:
                    key_to_check = store.keys()[0].replace('/', '')
                
                if key_to_check:
                    # select() allows loading a slice without loading the whole file
                    df_peek = store.select(key_to_check, start=0, stop=5)
                    print(f"   ✅ Access Successful! Table '{key_to_check}'")
                    print(f"   📊 Columns: {list(df_peek.columns)}")
                    print(f"   👀 First row preview:\n{df_peek.iloc[0].to_dict()}")
                else:
                    print(f"   ⚠️ No recognizable data tables found.")
                    
        except ImportError:
            print("❌ Error: Missing library. Run 'pip install tables'")
        except Exception as e:
            print(f"   ❌ Error reading {file_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 Bee Datasets")
    # Default to a local data folder
    parser.add_argument("--path", type=str, default="data/beetogether", help="Path to HDF5 files")
    args = parser.parse_args()
    
    inspect_datasets(args.path)
