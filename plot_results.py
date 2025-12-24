# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("results/per_frame_counts.csv")

# Convert time to minutes for readability
df["time_min"] = df["timestamp_s"] / 60.0

# ---------- Plot 1: Total vehicle count vs time ----------
plt.figure()
plt.plot(df["time_min"], df["total_count"])
plt.xlabel("Time (minutes)")
plt.ylabel("Total Vehicle Count")
plt.title("Total Vehicle Count vs Time")
plt.grid(True)
plt.savefig("results/plot_total_count.png", dpi=300)
plt.close()

# ---------- Plot 2: Density vs time ----------
plt.figure()
plt.plot(df["time_min"], df["density"])
plt.xlabel("Time (minutes)")
plt.ylabel("Traffic Density")
plt.title("Traffic Density vs Time")
plt.grid(True)
plt.savefig("results/plot_density.png", dpi=300)
plt.close()

# ---------- Plot 3: Per-class counts ----------
plt.figure()
plt.plot(df["time_min"], df["count_car"], label="Car")
plt.plot(df["time_min"], df["count_bus"], label="Bus")
plt.plot(df["time_min"], df["count_2w"], label="Two-Wheeler")
plt.plot(df["time_min"], df["count_3w"], label="Three-Wheeler")
plt.xlabel("Time (minutes)")
plt.ylabel("Vehicle Count")
plt.title("Per-Class Vehicle Counts vs Time")
plt.legend()
plt.grid(True)
plt.savefig("results/plot_per_class.png", dpi=300)
plt.close()

print("✅ Graphs saved in results/")
