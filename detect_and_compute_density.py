# detect_and_compute_density.py
import os
import cv2
import math
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from datetime import timedelta

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video file or frames folder")
    p.add_argument("--outdir", default="./results", help="output folder")
    p.add_argument("--model_name", required=True, help="local UVH-26 model path")
    p.add_argument("--window_seconds", type=int, default=15)
    p.add_argument("--fps_override", type=float, default=0)
    p.add_argument("--capacity", type=int, default=60)
    return p.parse_args()

# ✅ UVH-26 semantic mapping
VEHICLE_CLASSES = {
    # Cars
    "hatchback": "car",
    "sedan": "car",
    "suv": "car",
    "muv": "car",
    "van": "car",

    # Bus
    "bus": "bus",
    "mini-bus": "bus",
    "tempo-traveller": "bus",

    # Truck
    "truck": "truck",
    "lcv": "truck",

    # Two wheelers
    "two-wheeler": "2w",
    "bicycle": "2w",

    # Three wheelers
    "three-wheeler": "3w",

    # Others
    "others": "other"
}

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def compute_density(count, capacity):
    return min(1.0, count / max(1, capacity))

def counts_from_results(results):
    counts = {"car":0, "bus":0, "truck":0, "2w":0, "3w":0, "other":0}

    if results is None or results.boxes is None:
        return counts

    for b in results.boxes:
        cls_name = results.names[int(b.cls)].lower().strip()
        key = VEHICLE_CLASSES.get(cls_name, "other")
        counts[key] += 1

    counts["total"] = (
        counts["car"] +
        counts["bus"] +
        counts["truck"] +
        counts["2w"] +
        counts["3w"]
    )
    return counts

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "samples"))
    ensure_dir(os.path.join(args.outdir, "density_jsons"))

    print("Loading model:", args.model_name)
    model = YOLO(args.model_name)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if args.fps_override:
        fps = args.fps_override

    window_frames = int(args.window_seconds * fps)
    print(f"FPS={fps:.2f}, Window frames={window_frames}")

    per_frame_records = []
    window_records = []
    cur_window_start = 0
    window_idx = 0

    frame_idx = 0
    pbar = tqdm(desc="Frames processed")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]
        counts = counts_from_results(res)
        density = compute_density(counts["total"], args.capacity)

        ts_seconds = frame_idx / fps
        ts_hms = str(timedelta(seconds=int(ts_seconds)))

        per_frame_records.append({
            "frame_index": frame_idx,
            "timestamp_s": ts_seconds,
            "timestamp_hms": ts_hms,
            "count_car": counts["car"],
            "count_bus": counts["bus"],
            "count_truck": counts["truck"],
            "count_2w": counts["2w"],
            "count_3w": counts["3w"],
            "count_other": counts["other"],
            "total_count": counts["total"],
            "density": density
        })

        # Window aggregation
        if (frame_idx - cur_window_start + 1) >= window_frames:
            win = per_frame_records[-window_frames:]
            avg_count = np.mean([r["total_count"] for r in win])
            avg_density = np.mean([r["density"] for r in win])

            row = {
                "window_index": window_idx,
                "frame_start": cur_window_start,
                "frame_end": frame_idx,
                "avg_count": avg_count,
                "avg_density": avg_density
            }

            window_records.append(row)

            with open(os.path.join(args.outdir, "density_jsons", f"window_{window_idx:04d}.json"), "w") as f:
                json.dump(row, f, indent=2)

            cur_window_start = frame_idx + 1
            window_idx += 1

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Save CSVs
    pd.DataFrame(per_frame_records).to_csv(
        os.path.join(args.outdir, "per_frame_counts.csv"), index=False
    )
    pd.DataFrame(window_records).to_csv(
        os.path.join(args.outdir, "window_summaries.csv"), index=False
    )

    print("✅ Processing complete.")
    print("Saved per_frame_counts.csv and window_summaries.csv")

if __name__ == "__main__":
    main()
