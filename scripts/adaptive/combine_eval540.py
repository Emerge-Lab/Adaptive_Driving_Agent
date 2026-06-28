"""Combine the 60-cell eval (eval540_grid + eval540) into one wide CSV.

One row per (entropy_ub, k, seed, wid, map_id) with per-trial success rates
t0..t5 (empty beyond each cell's k) and ada_delta = t_last - t0. Entropy/partner
join mirrors analyze_entropy_k_adaptation.py exactly:
  eval540_grid -> partner+entropy via final_runs_manifest.csv (wid lookup)
  eval540      -> partner 2e029h15, entropy_ub = 0.10 (the manifest gap)

-> outputs/eval540_combined/all_cells.csv
"""
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

GRID = Path("outputs/eval540_grid")
SOLO = Path("outputs/eval540")
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
OUT = Path("outputs/eval540_combined")
OUT_CSV = OUT / "all_cells.csv"

SOLO_PARTNER, SOLO_ENTROPY = "2e029h15", 0.10
MAXK = 6
TCOLS = [f"t{i}" for i in range(MAXK)]
FNAME_RE = re.compile(r"per_map_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv")


def manifest_lookup():
    """wid -> (partner, entropy_ub, k, seed) from the manifest (45 grid wids)."""
    m = {}
    for r in csv.DictReader(open(MANIFEST)):
        m[r["wandb_id"]] = (r["partner"], float(r["entropy_ub"]),
                            int(r["k"]), int(r["seed"]))
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    look = manifest_lookup()
    files = [(f, False) for f in sorted(glob.glob(str(GRID / "per_map_k*_seed*.csv")))]
    files += [(f, True) for f in sorted(glob.glob(str(SOLO / "per_map_k*_seed*.csv")))]

    rows_out = []
    cells = []          # (entropy, k, seed, wid, source, nmaps)
    cover = defaultdict(set)  # (entropy, k) -> {seeds}
    for f, is_solo in files:
        mt = FNAME_RE.search(Path(f).name)
        k, seed, wid = int(mt.group(1)), int(mt.group(2)), mt.group(3)
        if is_solo:
            partner, ent = SOLO_PARTNER, SOLO_ENTROPY
        else:
            assert wid in look, f"{wid} missing from manifest ({f})"
            partner, ent, mk, ms = look[wid]
            assert (mk, ms) == (k, seed), f"{wid}: manifest k/seed {(mk,ms)} != filename {(k,seed)}"

        recs = list(csv.DictReader(open(f)))
        tcols = sorted([c for c in recs[0] if re.fullmatch(r"t\d+", c)],
                       key=lambda c: int(c[1:]))
        assert len(tcols) == k, f"{f}: {len(tcols)} t-cols but k={k}"
        for r in recs:
            mid = int(float(r["map_id"]))
            tv = [float(r[c]) for c in tcols]
            ada = float(r["ada_delta_last_minus_0"])
            assert abs(ada - (tv[-1] - tv[0])) < 1e-6, f"{f} map {mid}: ada mismatch"
            padded = tv + [""] * (MAXK - k)
            rows_out.append([partner, ent, k, seed, wid, mid, *padded, ada])
        cells.append((ent, k, seed, wid, "solo" if is_solo else "grid", len(recs)))
        cover[(ent, k)].add(seed)

    # ---- validation ----
    assert len(cells) == 60, f"{len(cells)} cells, expected 60"
    nmaps = {c[5] for c in cells}
    assert len(nmaps) == 1, f"cells disagree on map count: {sorted(nmaps)}"
    for (ent, k), seeds in sorted(cover.items()):
        assert seeds == {42, 43, 44}, f"entropy={ent} k={k}: seeds {sorted(seeds)}"
    assert len(cover) == 20, f"{len(cover)} entropy×k combos, expected 20"

    rows_out.sort(key=lambda r: (r[1], r[2], r[3], r[5]))  # ent, k, seed, map_id
    header = ["partner", "entropy_ub", "k", "seed", "wid", "map_id", *TCOLS, "ada_delta"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows_out)

    n_maps = nmaps.pop()
    print(f"wrote {OUT_CSV}")
    print(f"  {len(cells)} cells (45 grid + 15 solo) x {n_maps} maps = {len(rows_out)} rows")
    print("  coverage (entropy x k, all should be {42,43,44}):")
    for ent in [0.05, 0.10, 0.20, 0.50]:
        ks = " ".join(f"k{k}:{sorted(cover[(ent,k)])}" for k in [2,3,4,5,6])
        print(f"    entropy={ent:.2f}  {ks}")


if __name__ == "__main__":
    main()
