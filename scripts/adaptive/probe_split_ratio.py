"""Inspect ratio split by active vs limbo at epoch 0 minibatch 0."""
import json
import sys
from collections import defaultdict


def inspect(path):
    inner_by_epoch = defaultdict(list)
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
                if r.get("event") == "gae_inner":
                    inner_by_epoch[r["epoch"]].append(r)
            except Exception:
                pass

    print(f"=== {path.split('/')[-1]} ===")
    for ep in sorted(inner_by_epoch.keys())[:2]:
        records = sorted(inner_by_epoch[ep], key=lambda r: r["minibatch"])
        for r0 in records[:3]:
            mb = r0.get("minibatch", 0)
            print(f"\n  epoch {ep} mb {mb}:")
            print(f"    OVERALL : mean={r0.get('ratio_mean', 0):8.4f} min={r0.get('ratio_min', 0):8.4f} max={r0.get('ratio_max', 0):8.4f}")
            print(f"    ACTIVE  : n={r0.get('active_n', 0):6d} mean={r0.get('active_ratio_mean', 0):8.4f} min={r0.get('active_ratio_min', 0):8.4f} max={r0.get('active_ratio_max', 0):8.4f} std={r0.get('active_ratio_std', 0):8.4f}")
            print(f"    LIMBO   : n={r0.get('limbo_n', 0):6d} mean={r0.get('limbo_ratio_mean', 0):8.4f} min={r0.get('limbo_ratio_min', 0):8.4f} max={r0.get('limbo_ratio_max', 0):8.4f} std={r0.get('limbo_ratio_std', 0):8.4f}")


if __name__ == "__main__":
    inspect(sys.argv[1])
