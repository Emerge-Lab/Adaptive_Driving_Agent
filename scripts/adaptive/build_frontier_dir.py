"""Build nuplan_201_frontier: maps the k=4/e=0.10 agent has NOT mastered
(mean zero-shot t0 < 0.9 across 3 seeds x 5 rollouts from the map-scoring
sweep), excluding nuplan_hard originals (eval set) and ids >= 4999 (unseen pool).

Output:
  resources/drive/binaries/nuplan_201_frontier/map_%03d.bin (renumbered symlinks)
  outputs/map_scoring/frontier_manifest.csv (local_id, orig_id, mean_t0)
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

SRC = Path("resources/drive/binaries/nuplan_201")
HARD_EVAL = Path("resources/drive/binaries/nuplan_hard")
OUT = Path("resources/drive/binaries/nuplan_201_frontier")
MANIFEST = Path("outputs/map_scoring/frontier_manifest.csv")
T0_MAX = 0.9
HELDOUT_START = 4999
WIDS = {"qxw6c0jh": 42, "ufmegw4l": 43, "jsckmpha": 44}


def main():
    c2o = {}
    for r in csv.DictReader(open("outputs/map_scoring/chunk_manifest.csv")):
        c2o[(int(r["chunk"]), int(r["local_id"]))] = int(r["orig_id"])

    t0 = defaultdict(list)
    for c in range(10):
        for wid, seed in WIDS.items():
            f = Path(f"outputs/map_scoring/chunk{c:02d}_{wid}/per_map_k4_seed{seed}_{wid}.csv")
            for r in csv.DictReader(open(f)):
                t0[c2o[(c, int(float(r["map_id"])))]].append(float(r["t0"]))

    eval_orig = {int(p.resolve().name.replace("map_", "").replace(".bin", ""))
                 for p in HARD_EVAL.glob("map_*.bin")}

    keep = []
    for oid, vals in sorted(t0.items()):
        if len(vals) < 3 or oid in eval_orig or oid >= HELDOUT_START:
            continue
        m = float(np.mean(vals))
        if m < T0_MAX:
            keep.append((oid, m))
    print(f"frontier maps: {len(keep)} (t0 < {T0_MAX}, eval + heldout excluded)")

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("map_*.bin"):
        old.unlink()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["local_id", "orig_id", "mean_t0"])
        for local_id, (oid, m) in enumerate(keep):
            (OUT / f"map_{local_id:03d}.bin").symlink_to((SRC / f"map_{oid:03d}.bin").resolve())
            w.writerow([local_id, oid, f"{m:.4f}"])
    print(f"built {OUT}; manifest {MANIFEST}")


if __name__ == "__main__":
    main()
