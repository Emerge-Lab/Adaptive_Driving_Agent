#!/usr/bin/env python3
"""Read per-agent success log from each /tmp/recovery_<wid>.json and print
conditional-success tables. Pure data — no judgment calls in the script."""

import json
import os

RUNS = [
    ("y9tges7d", "ampggcji  e=0.10 d=0.8"),
    ("bzzosxsg", "hntq6ykn  e=0.01 d=0.8"),
    ("5p6tl3pt", "q1mtzo02  e=0.10 d=0.6"),
    ("3p5ome2t", "ureqzfe9  e=0.01 d=0.6"),
]


def cond_rate(records, condition_fn, target_fn):
    """P(target | condition). Returns (rate, num_target, num_condition)."""
    cond = [r for r in records if condition_fn(r)]
    if not cond:
        return None, 0, 0
    target = [r for r in cond if target_fn(r)]
    return len(target) / len(cond), len(target), len(cond)


def fmt(rate, num, den):
    if rate is None:
        return "—"
    return f"{100 * rate:5.1f}% ({num}/{den})"


for wid, label in RUNS:
    path = f"/tmp/recovery_{wid}.json"
    if not os.path.exists(path):
        print(f"\n## {label} ({wid})\n  NO DATA")
        continue
    d = json.load(open(path))
    records = d.get("per_agent_success_log", [])
    if not records:
        print(f"\n## {label} ({wid})\n  NO LOG")
        continue
    k = max(int(k[1:]) for k in records[0].keys() if k.startswith("s")) + 1
    n = len(records)

    print(f"\n## {label} ({wid})  — {n} (rollout, agent) cells, k={k} scenarios")

    # Marginals
    print(f"  marginals:")
    for s in range(k):
        r, t, dn = cond_rate(records, lambda x: True, lambda x, s=s: x[f"s{s}"] == 1)
        print(f"    s{s} succeed: {fmt(r, t, dn)}")

    # Conditionals: P(succeed s_target | fail s_cond) and P(succeed s_target | succeed s_cond)
    print(f"  conditional success — given EARLIER FAILURE:")
    for s_cond in range(k - 1):
        for s_target in range(s_cond + 1, k):
            r, t, dn = cond_rate(
                records,
                lambda x, s_cond=s_cond: x[f"s{s_cond}"] == 0,
                lambda x, s_target=s_target: x[f"s{s_target}"] == 1,
            )
            print(f"    P(succeed s{s_target} | fail s{s_cond}) = {fmt(r, t, dn)}")

    print(f"  conditional success — given EARLIER SUCCESS:")
    for s_cond in range(k - 1):
        for s_target in range(s_cond + 1, k):
            r, t, dn = cond_rate(
                records,
                lambda x, s_cond=s_cond: x[f"s{s_cond}"] == 1,
                lambda x, s_target=s_target: x[f"s{s_target}"] == 1,
            )
            print(f"    P(succeed s{s_target} | succeed s{s_cond}) = {fmt(r, t, dn)}")
