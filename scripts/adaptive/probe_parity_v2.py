"""Check ratio at epoch 0 minibatch 0 of a specific jsonl."""
import json
import sys
from collections import defaultdict


def inspect(path, label):
    inner_by_epoch = defaultdict(list)
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
                if r.get("event") == "gae_inner":
                    inner_by_epoch[r["epoch"]].append(r)
            except Exception:
                pass

    print(f"\n=== {label} ({path.split('/')[-1]}) ===")
    print(f"  Format: epoch  mb   ratio_mean  ratio_min  ratio_max  approx_kl  clipfrac")
    for ep in sorted(inner_by_epoch.keys())[:3]:
        records = sorted(inner_by_epoch[ep], key=lambda r: r["minibatch"])
        for r0 in records[:3]:
            ratio_mean = r0.get("ratio_mean", 0)
            ratio_min = r0.get("ratio_min", 0)
            ratio_max = r0.get("ratio_max", 0)
            kl = r0.get("approx_kl", 0)
            cf = r0.get("clipfrac", 0)
            mb = r0.get("minibatch", 0)
            print(
                f"    {ep:3d}  {mb:3d}  "
                f"{ratio_mean:11.6f}  {ratio_min:9.6f}  {ratio_max:9.6f}  "
                f"{kl:9.6f}  {cf:8.4f}"
            )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p, p.split("/")[-1])
