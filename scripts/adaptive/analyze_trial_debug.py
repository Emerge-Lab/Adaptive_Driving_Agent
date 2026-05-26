"""Diff two trial-debug jsonl files (gb=0 vs gb=3) — summary stats per event type.

Usage: python analyze_trial_debug.py <gb0.jsonl> <gb3.jsonl>
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    events = defaultdict(list)
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
                events[r["event"]].append(r)
            except Exception:
                pass
    return events


def stats(records, key):
    vals = [float(r.get(key, 0) or 0) for r in records if key in r]
    if not vals:
        return None
    arr = np.array(vals)
    return (
        f"n={len(arr):4d}  mean={arr.mean():12.4f}  std={arr.std():12.4f}  "
        f"min={arr.min():12.4f}  max={arr.max():12.4f}"
    )


def compare(a, b, ev, keys):
    print(f"\n=== {ev} ===")
    for k in keys:
        s_a = stats(a[ev], k)
        s_b = stats(b[ev], k)
        if s_a or s_b:
            print(f"  {k}")
            print(f"    gb0: {s_a}")
            print(f"    gb3: {s_b}")


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = load(a_path)
    b = load(b_path)

    print(f"event counts gb0: {[(k, len(v)) for k, v in a.items()]}")
    print(f"event counts gb3: {[(k, len(v)) for k, v in b.items()]}")

    compare(
        a, b, "gae_outer_pre",
        [
            "terminals_sum", "truncations_sum",
            "bootstrap_stop_sum", "bootstrap_overlap",
            "values_mean", "values_std",
            "rewards_mean", "rewards_sum",
        ],
    )

    compare(
        a, b, "gae_outer_post",
        [
            "adv_mean", "adv_std", "adv_min", "adv_max",
            "adv_nan_count", "adv_inf_count",
        ],
    )

    compare(
        a, b, "epoch_end",
        [
            "score", "explained_var", "value_loss", "policy_loss",
            "entropy", "approx_kl", "clipfrac",
        ],
    )

    # Cache reset rate
    print("\n=== cache_reset ===")
    print(f"  gb0: {len(a.get('cache_reset', []))} events")
    print(f"  gb3: {len(b.get('cache_reset', []))} events")
    if a.get("cache_reset"):
        ds = [r.get("done_mask_sum", 0) for r in a["cache_reset"]]
        print(f"    gb0 done_mask_sum  mean={np.mean(ds):.1f}  total={sum(ds)}")
    if b.get("cache_reset"):
        ds = [r.get("done_mask_sum", 0) for r in b["cache_reset"]]
        print(f"    gb3 done_mask_sum  mean={np.mean(ds):.1f}  total={sum(ds)}")


if __name__ == "__main__":
    main()
