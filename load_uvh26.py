# load_uvh26.py
# Usage: python load_uvh26.py --split train --outdir ./uvh26_train
import os, argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--split', choices=['train','val','all'], default='train')
parser.add_argument('--outdir', default='./uvh26')
args = parser.parse_args()

if args.split == 'train':
    ds = load_dataset("iisc-aim/UVH-26", split="train")
elif args.split == 'val':
    ds = load_dataset("iisc-aim/UVH-26", split="validation")
else:
    ds = load_dataset("iisc-aim/UVH-26")

print("Loaded dataset, length:", len(ds))
os.makedirs(args.outdir, exist_ok=True)
print("NOTE: the dataset already includes COCO JSON files in the Hugging Face repo; this helper is just to inspect entries.")
print("Example dataset fields:", ds.column_names if hasattr(ds,'column_names') else ds)
# don't force-download images here if you prefer to use COCO JSON directly from HF web UI
