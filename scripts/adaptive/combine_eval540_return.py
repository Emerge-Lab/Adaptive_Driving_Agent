"""Combine the 60 return CSVs (outputs/eval540_return/per_map_R_*.csv) into one
wide table. Adapted from combine_eval540.py with the same wid→entropy join:
  - 45 grid wids in scripts/adaptive/final_runs_manifest.csv → partner+entropy_ub
  - 15 solo wids (2e029h15 partner) NOT in manifest → entropy_ub = 0.10

Wide table, one row per (entropy_ub, k, seed, wid, map_id):
  partner, entropy_ub, k, seed, wid, map_id, t0..t5 (empty beyond k), ada_delta_R

-> outputs/eval540_return/all_cells_R.csv
"""
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

IN = Path("outputs/eval540_return")
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
OUT_CSV = IN / "all_cells_R.csv"

SOLO_PARTNER, SOLO_ENTROPY = "2e029h15", 0.10
MAXK = 6
TCOLS = [f"t{i}" for i in range(MAXK)]
FNAME_RE = re.compile(r"per_map_R_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv")


def manifest_lookup():
    """wid -> (partner, entropy_ub, k, seed) from the 45-row manifest."""
    m = {}
    for r in csv.DictReader(open(MANIFEST)):
        m[r["wandb_id"]] = (r["partner"], float(r["entropy_ub"]),
                            int(r["k"]), int(r["seed"]))
    return m


def main():
    look = manifest_lookup()
    files = sorted(glob.glob(str(IN / "per_map_R_k*_seed*.csv")))

    rows_out = []
    cells = []          # (entropy, k, seed, wid, nmaps)
    cover = defaultdict(set)  # (entropy, k) -> {seeds}
    for f in files:
        mt = FNAME_RE.search(Path(f).name)
        if mt is None:
            continue
        k, seed, wid = int(mt.group(1)), int(mt.group(2)), mt.group(3)
        if wid in look:
            partner, ent, mk, ms = look[wid]
            assert (mk, ms) == (k, seed), f"{wid}: manifest k/seed {(mk,ms)} != filename {(k,seed)}"
        else:
            partner, ent = SOLO_PARTNER, SOLO_ENTROPY

        recs = list(csv.DictReader(open(f)))
        tcols = sorted([c for c in recs[0] if re.fullmatch(r"t\d+", c)],
                       key=lambda c: int(c[1:]))
        assert len(tcols) == k, f"{f}: {len(tcols)} t-cols but k={k}"
        for r in recs:
            mid = int(float(r["map_id"]))
            tv = [float(r[c]) for c in tcols]
            ada = float(r["ada_delta_R_last_minus_0"])
            assert abs(ada - (tv[-1] - tv[0])) < 1e-3, f"{f} map {mid}: ada_R mismatch"
            padded = tv + [""] * (MAXK - k)
            rows_out.append([partner, ent, k, seed, wid, mid, *padded, ada])
        cells.append((ent, k, seed, wid, len(recs)))
        cover[(ent, k)].add(seed)

    # ---- validation ----
    assert len(cells) == 60, f"{len(cells)} cells, expected 60"
    nmaps = {c[4] for c in cells}
    assert len(nmaps) == 1, f"cells disagree on map count: {sorted(nmaps)}"
    for (ent, k), seeds in sorted(cover.items()):
        assert seeds == {42, 43, 44}, f"entropy={ent} k={k}: seeds {sorted(seeds)}"
    assert len(cover) == 20, f"{len(cover)} entropy×k combos, expected 20"

    rows_out.sort(key=lambda r: (r[1], r[2], r[3], r[5]))
    header = ["partner", "entropy_ub", "k", "seed", "wid", "map_id", *TCOLS, "ada_delta_R"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows_out)

    n_maps = nmaps.pop()
    print(f"wrote {OUT_CSV}")
    print(f"  {len(cells)} cells x {n_maps} maps = {len(rows_out)} rows")
    print("  coverage (entropy x k, all {42,43,44}):")
    for ent in [0.05, 0.10, 0.20, 0.50]:
        ks = " ".join(f"k{k}:{sorted(cover[(ent,k)])}" for k in [2,3,4,5,6])
        print(f"    entropy={ent:.2f}  {ks}")


if __name__ == "__main__":
    main()
