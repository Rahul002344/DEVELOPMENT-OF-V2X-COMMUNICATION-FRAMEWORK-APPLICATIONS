# download_uvh.py
from huggingface_hub import snapshot_download
import os, shutil, sys

repo_id = "iisc-aim/UVH-26"
print("Downloading UVH-26 snapshot from Hugging Face (this may take a minute)...")
repo_path = snapshot_download(repo_id=repo_id, repo_type="model")
print("Snapshot downloaded to:", repo_path)

candidates = []
for root, dirs, files in os.walk(repo_path):
    for f in files:
        if f.lower().endswith((".pt", ".safetensors")):
            candidates.append(os.path.join(root, f))

if not candidates:
    print("No .pt or .safetensors files found in the repo snapshot. Exiting.")
    sys.exit(1)

# copy the first candidate to D:\Major\models\uvh26.pt
dst = r"D:\Major\models\uvh26.pt"
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(candidates[0], dst)
print("Copied", candidates[0], "->", dst)
print("Done.")
