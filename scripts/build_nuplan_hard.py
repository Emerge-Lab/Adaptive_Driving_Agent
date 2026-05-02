"""Create nuplan_hard binary directory: top 10% maps by SDC interaction steps.

Reads /tmp/nuplan_201_hardness_scores.csv, sorts by sdc_interaction_steps desc,
takes top 10% (~540 maps), and creates symlinks in
resources/drive/binaries/nuplan_hard/ pointing at the original .bin files
in resources/drive/binaries/nuplan_201/.

The new bin files are renumbered sequentially (map_001.bin, map_002.bin, ...)
so num_maps in the eval config matches the directory count.
"""
import csv, os, sys, json, argparse, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="/tmp/nuplan_201_hardness_scores.csv")
ap.add_argument("--source-dir", default="/workspace/ADA/resources/drive/binaries/nuplan_201")
ap.add_argument("--out-dir", default="/workspace/ADA/resources/drive/binaries/nuplan_hard")
ap.add_argument("--metric", default="sdc_interaction_steps",
                choices=["sdc_interaction_steps", "interaction_events"])
ap.add_argument("--top-pct", type=float, default=10.0)
args = ap.parse_args()

# Load scores
rows = []
with open(args.scores) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append({
            "bin_id": int(r["bin_id"]),
            "scenario_id": r["scenario_id"],
            "metric": int(r[args.metric]),
            "num_valid_vehicles": int(r["num_valid_vehicles"]),
            "total_steps": int(r["total_steps"]),
        })
print(f"Loaded {len(rows)} maps from {args.scores}")

# Sort by metric desc
rows.sort(key=lambda r: -r["metric"])
n_top = max(1, int(len(rows) * args.top_pct / 100))
top = rows[:n_top]
threshold = top[-1]["metric"]
print(f"Top {args.top_pct}% = {n_top} maps. Threshold: {args.metric} >= {threshold}")
print(f"  highest score: {top[0]['metric']} ({top[0]['scenario_id']})")
print(f"  lowest in hard set: {top[-1]['metric']} ({top[-1]['scenario_id']})")

# Build out dir
out_dir = Path(args.out_dir)
if out_dir.exists():
    print(f"  removing existing {out_dir}")
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)

# Sort the hard set by original bin_id so the renumbering preserves a
# deterministic order. (Sort by score determines membership, then we sort
# by bin_id for stable filename assignment.)
top.sort(key=lambda r: r["bin_id"])

src_dir = Path(args.source_dir)
n_linked = 0
n_missing = 0
for new_idx, r in enumerate(top, start=1):
    # Source bins use 3+digit padding; existing files match map_001..map_5401
    src_name = f"map_{r['bin_id']:03d}.bin"
    src_path = src_dir / src_name
    if not src_path.exists():
        if n_missing < 5:
            print(f"  MISSING: {src_path}")
        n_missing += 1
        continue
    dst_name = f"map_{new_idx:03d}.bin"
    dst_path = out_dir / dst_name
    os.symlink(src_path.resolve(), dst_path)
    n_linked += 1

print(f"\nLinked {n_linked} maps into {out_dir}")
if n_missing > 0:
    print(f"  WARNING: {n_missing} bin files missing from source dir")

# Save manifest mapping new_idx -> (orig_bin_id, scenario_id, metric_value)
manifest_path = out_dir / "_manifest.csv"
with open(manifest_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["new_bin_id", "orig_bin_id", "scenario_id", args.metric, "num_valid_vehicles"])
    for new_idx, r in enumerate(top, start=1):
        if (src_dir / f"map_{r['bin_id']:03d}.bin").exists():
            w.writerow([new_idx, r["bin_id"], r["scenario_id"], r["metric"], r["num_valid_vehicles"]])

print(f"Manifest: {manifest_path}")

# Stats
metrics = [r["metric"] for r in top if (src_dir / f"map_{r['bin_id']:03d}.bin").exists()]
print(f"\nHard set stats:")
print(f"  count: {len(metrics)}")
print(f"  {args.metric} mean: {sum(metrics)/len(metrics):.1f}")
print(f"  {args.metric} min:  {min(metrics)}")
print(f"  {args.metric} max:  {max(metrics)}")
