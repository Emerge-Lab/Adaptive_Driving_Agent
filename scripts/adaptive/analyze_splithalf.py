"""Split-half selector re-analysis (reviewer risk #2 in paper_analysis.md).

Concern: if the adaptable-map selector (p0 < 0.8 on trial-0 success) uses the
same rollouts as the dR = t_last - t0 measurement, regression-to-the-mean can
inflate dR. Here both are computed from ONE eval's raw per-(map, rollout)
records (outputs/eval540_splithalf/, written by eval_final_540.py
--dump-per-rollout) on DISJOINT rollout halves:

  - select-A/measure-B: p0 from rollouts 0-9, per-trial return from 10-19
  - select-B/measure-A: the reverse (robustness)
  - same-half (biased upper reference): select and measure on the same half
  - full-sample: select and measure on all 20 rollouts (what the headline does)

If the disjoint-half dR stays close to the full-sample dR, the headline
selector is bias-clean. Usage: python scripts/adaptive/analyze_splithalf.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "outputs" / "eval540_splithalf"
TRIALS = ["t0", "t1", "t2", "t3"]
P0_THRESH = 0.8

CELLS = {
    "0.10/k4": {"qxw6c0jh": 42, "ufmegw4l": 43, "jsckmpha": 44},
    "0.20/k4": {"ftxa55g3": 42, "citbzhdc": 43, "c0k9uqhc": 44},
}
HALF_A = list(range(0, 10))
HALF_B = list(range(10, 20))
FULL = list(range(20))


def load(kind, wid, seed):
    df = pd.read_csv(D / f"per_rollout_{kind}_k4_seed{seed}_{wid}.csv")
    return df


def cell_analysis(name, wids):
    succ = {w: load("success", w, s) for w, s in wids.items()}
    ret = {w: load("R", w, s) for w, s in wids.items()}

    def selector(rollouts):
        """adaptable map set: mean t0 success over given rollouts, avg seeds"""
        p0 = pd.concat([
            s[s.rollout.isin(rollouts)].groupby("map_id").t0.mean()
            for s in succ.values()
        ], axis=1).mean(axis=1)
        return set(p0[p0 < P0_THRESH].index)

    def measure(maps, rollouts):
        """per-seed mean per-trial return on maps x rollouts -> (mean dR, per-seed)"""
        per_seed = []
        for r in ret.values():
            sub = r[r.rollout.isin(rollouts) & r.map_id.isin(maps)]
            per_seed.append(sub[TRIALS].mean())
        m = pd.concat(per_seed, axis=1)
        dR = (m.loc["t3"] - m.loc["t0"]).values
        return m.mean(axis=1), dR

    rows = []
    for label, sel_r, meas_r in [
        ("full-sample (headline-style)", FULL, FULL),
        ("select A / measure B (clean)", HALF_A, HALF_B),
        ("select B / measure A (clean)", HALF_B, HALF_A),
        ("same-half A (biased ref)", HALF_A, HALF_A),
        ("same-half B (biased ref)", HALF_B, HALF_B),
    ]:
        maps = selector(sel_r)
        curve, dR = measure(maps, meas_r)
        rows.append(dict(cell=name, scheme=label, n_maps=len(maps),
                         **{t: round(float(curve[t]), 3) for t in TRIALS},
                         dR_mean=round(float(np.mean(dR)), 3),
                         dR_per_seed=[round(float(x), 2) for x in dR]))
    return rows


def main():
    all_rows = []
    for name, wids in CELLS.items():
        all_rows += cell_analysis(name, wids)
    out = pd.DataFrame(all_rows)
    out.to_csv(D / "splithalf_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nwrote {D}/splithalf_summary.csv")


if __name__ == "__main__":
    main()
