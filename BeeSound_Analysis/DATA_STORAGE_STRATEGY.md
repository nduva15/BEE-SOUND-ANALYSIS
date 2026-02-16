# 📋 PRD: Data Storage Infrastructure (BeeSound Analysis)

**Version:** 1.0  
**Status:** Approved  
**Dataset Size:** 28GB+ (High Resolution Audio)

## 1. Problem Statement
The BeeSound Analysis dataset (NUHIVE, TBON, etc.) totals **28GB**, which exceeds the capacity/performance of standard local development machines. We need a remote storage solution that is **scalable**, **accessible**, and **cost-effective**.

## 2. Strategic Recommendation
**do NOT use Hostinger.** It is for hosting websites (HTML/PHP), not "Big Data."

### ✅ The Selected Solution: **Kaggle Datasets (Private or Public)**
We will upload the data to Kaggle. It acts as our free, high-speed cloud storage.

*   **Cost:** $0.00
*   **Storage Limit:** ~100GB per dataset (Perfect for us).
*   **API:** Python accessible.
*   **Compute:** You can run the training *on Kaggle's GPU* instead of your laptop.

---

## 3. Implementation Plan

### 3.1 Architecture: "Lazy Loading"
Instead of downloading 28GB to your laptop, we will configure the project to:
1.  **Index** the remote files (Metadata only).
2.  **Download** only the specific samples needed for the current test (e.g., "Download just the Queen Piping events").

### 3.2 Workflow
1.  **Upload**: Upload the `.h5` files to a reusable Kaggle Dataset.
2.  **API Config**: Place `kaggle.json` (API Key) in `~/.kaggle/`.
3.  **Fetch Script**: Use the python script below to fetch data on demand.

---

## 4. Tooling: The Data Fetcher (kaggle_loader.py)
This script allows you to download *just* the file you need, preventing a disk crash.

```python
# To be implemented in tools/kaggle_loader.py
import kaggle
import os

def fetch_dataset(dataset_name="beetogether-audio", file_name=None):
    """
    Downloads BeeSound data from cloud storage.
    If file_name is None, downloads everything (Warning: 28GB).
    """
    target_dir = "data/raw_audio/"
    
    print(f"⬇️ Downloading {file_name or 'ALL'} from Kaggle...")
    kaggle.api.dataset_download_files(
        f"nduva15/{dataset_name}", # Replace with actual dataset ID
        path=target_dir, 
        unzip=True,
        quiet=False
    )
    print("✅ Download Complete.")
```

## 5. Alternative: Wasabi (S3 Compatible)
If the data must remain strict corporate private:
*   **Provider:** Wasabi Hot Cloud Storage.
*   **Cost:** ~$5.99/month (1TB min).
*   **Tool:** `boto3` (Python S3 library).

---

## 6. Decision Record
**Decision:** Proceed with **Kaggle Datasets** for hosting.
**Reason:** It solves the storage issue (28GB free) AND the compute issue (free GPUs for training).
