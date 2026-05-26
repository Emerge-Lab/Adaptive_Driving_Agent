"""Quick health check: per-epoch summary of clipfrac, approx_kl, explained_var,
ratio stats from a trial_debug jsonl."""
import json
import sys
from collections import defaultdict

import numpy as np


def main():
    path = sys.argv[1]
    inner = defaultdict(list)
    epoch_end = defaultdict(list)
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
                e = r.get("event")
                if e == "gae_inner":
                    inner[r["epoch"]].append(r)
                elif e == "epoch_end":
                    epoch_end[r["epoch"]].append(r)
            except Exception:
                pass

    print(f"=== {path.split('/')[-1]} ===")
    print(f"{'ep':>3}  {'cf':>7}  {'kl':>7}  {'expvar':>8}  {'active_r_mean':>14}  {'active_r_std':>13}  {'active_r_max':>13}")
    for ep in sorted(inner.keys()):
        recs = inner[ep]
        cfs = [r.get("clipfrac", 0) for r in recs]
        kls = [r.get("approx_kl", 0) for r in recs]
        amns = [r.get("active_ratio_mean", 0) for r in recs]
        astds = [r.get("active_ratio_std", 0) for r in recs]
        amxs = [r.get("active_ratio_max", 0) for r in recs]
        ee = epoch_end.get(ep, [{}])
        ev = ee[0].get("explained_var", 0) if ee else 0
        print(f"{ep:>3}  {np.mean(cfs):>7.4f}  {np.mean(kls):>7.4f}  {ev:>8.4f}  {np.mean(amns):>14.4f}  {np.mean(astds):>13.4f}  {np.max(amxs):>13.4f}")


if __name__ == "__main__":
    main()
