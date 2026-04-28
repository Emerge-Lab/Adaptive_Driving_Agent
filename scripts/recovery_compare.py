#!/usr/bin/env python3
"""Compare adaptive eval (cache preserved) vs control eval (cache reset).
The lift is the actual adaptation signal."""
import json
import os

RUNS = [
    ("y9tges7d", "ampggcji  e=0.10 d=0.8"),
    ("bzzosxsg", "hntq6ykn  e=0.01 d=0.8"),
    ("5p6tl3pt", "q1mtzo02  e=0.10 d=0.6"),
    ("3p5ome2t", "ureqzfe9  e=0.01 d=0.6"),
]


def load(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return json.load(open(path)).get("per_agent_success_log", [])


def cond_rate(records, cond_fn, target_fn):
    cond = [r for r in records if cond_fn(r)]
    if not cond:
        return None, 0, 0
    target = [r for r in cond if target_fn(r)]
    return len(target) / len(cond), len(target), len(cond)


def fmt(rate, num, den):
    if rate is None:
        return "       —"
    return f"{100*rate:5.1f}% ({num:>4}/{den})"


print("=" * 110)
print("  ADAPTIVE (cache preserved across scenarios) vs CONTROL (cache reset every scenario)")
print("  If 'lift' is positive, the K/V cache is helping recover hard cases.")
print("=" * 110)

header = f"{'co-player':<28} {'metric':<28} {'adaptive':<22} {'control':<22} {'lift':<10}"
print(header)
print("-" * 110)

for wid, label in RUNS:
    adapt = load(f"/tmp/recovery_{wid}.json")
    ctrl = load(f"/tmp/recovery_control_{wid}.json")
    if adapt is None or ctrl is None:
        print(f"{label:<28} (missing data)")
        continue
    k = max(int(k[1:]) for k in adapt[0].keys() if k.startswith("s")) + 1

    # Conditional success P(succeed s_K | fail s_0) is the headline
    metrics = [
        ("P(succ s1 | fail s0)", lambda r: r["s0"] == 0, lambda r: r.get("s1", 0) == 1),
        ("P(succ s2 | fail s0)", lambda r: r["s0"] == 0, lambda r: r.get("s2", 0) == 1),
        ("P(succ s2 | fail s1)", lambda r: r["s1"] == 0, lambda r: r.get("s2", 0) == 1),
        ("P(succ s2 | succ s0)", lambda r: r["s0"] == 1, lambda r: r.get("s2", 0) == 1),
        ("marginal s0", lambda r: True, lambda r: r["s0"] == 1),
        ("marginal s2", lambda r: True, lambda r: r.get("s2", 0) == 1),
    ]

    print(f"{label:<28}", end="")
    first = True
    for name, cond, tgt in metrics:
        a_r, a_t, a_d = cond_rate(adapt, cond, tgt)
        c_r, c_t, c_d = cond_rate(ctrl, cond, tgt)
        if a_r is None or c_r is None:
            lift_str = "—"
        else:
            lift = a_r - c_r
            lift_str = f"{lift:+.3f}"
        if first:
            print(f" {name:<28} {fmt(a_r,a_t,a_d):<22} {fmt(c_r,c_t,c_d):<22} {lift_str:<10}")
            first = False
        else:
            print(f"{'':<28} {name:<28} {fmt(a_r,a_t,a_d):<22} {fmt(c_r,c_t,c_d):<22} {lift_str:<10}")
    print()
