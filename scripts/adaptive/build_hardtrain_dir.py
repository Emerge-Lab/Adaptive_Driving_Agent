"""Build nuplan_201_hardtrain: interaction-dense training maps for the
curriculum continuation experiment.

Selection: sdc_interaction_steps >= THRESHOLD (top ~25% of nuplan_201),
excluding (a) the 540 nuplan_hard originals (eval set — a hardness filter
would otherwise concentrate training on eval maps), (b) original ids >= 4999
(never sampled in training; preserved as the unseen confirmation pool).

Output:
  resources/drive/binaries/nuplan_201_hardtrain/map_%03d.bin (renumbered symlinks)
  outputs/map_scoring/hardtrain_manifest.csv (local_id, orig_id, sdc_interaction_steps)
"""
import csv
from pathlib import Path

SCORES = Path("scripts/nuplan_201_hardness_scores.csv")
SRC = Path("resources/drive/binaries/nuplan_201")
HARD_EVAL = Path("resources/drive/binaries/nuplan_hard")
OUT = Path("resources/drive/binaries/nuplan_201_hardtrain")
MANIFEST = Path("outputs/map_scoring/hardtrain_manifest.csv")
THRESHOLD = 21     # sdc_interaction_steps — top ~25%
HELDOUT_START = 4999


def main():
    eval_orig = set()
    for p in HARD_EVAL.glob("map_*.bin"):
        eval_orig.add(int(p.resolve().name.replace("map_", "").replace(".bin", "")))
    print(f"eval originals excluded: {len(eval_orig)}")

    keep = []
    for r in csv.DictReader(open(SCORES)):
        oid = int(r["bin_id"])
        steps = int(r["sdc_interaction_steps"])
        if steps < THRESHOLD or oid in eval_orig or oid >= HELDOUT_START:
            continue
        keep.append((oid, steps))
    keep.sort()
    print(f"selected: {len(keep)} maps (threshold >= {THRESHOLD})")

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("map_*.bin"):
        old.unlink()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["local_id", "orig_id", "sdc_interaction_steps"])
        for local_id, (oid, steps) in enumerate(keep):
            dst = OUT / f"map_{local_id:03d}.bin"
            dst.symlink_to((SRC / f"map_{oid:03d}.bin").resolve())
            w.writerow([local_id, oid, steps])
    print(f"built {OUT} ({len(keep)} maps); manifest {MANIFEST}")


if __name__ == "__main__":
    main()
