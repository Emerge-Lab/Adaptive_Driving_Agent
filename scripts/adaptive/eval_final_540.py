"""Offline human-replay eval of ONE final checkpoint at full scale (540 maps,
20 rollouts), generalized over k. Mirrors the in-line training eval
(pufferlib.utils.run_human_replay_eval_in_subprocess) so numbers are
apples-to-apples across k2..k6.

For the given (wid, k, iter) it:
  1. spawns `pufferl eval` against the checkpoint (k-scenarios=k, horizon=k*sl),
  2. parses the HUMAN_REPLAY_METRICS JSON block (incl. per_agent_success_log),
  3. writes a local per-map CSV (map_id, t0..t_{k-1}, ada_delta_last_minus_0) in
     the same naming the plot scripts expect, and
  4. logs the per_map_summary wandb.Table + scalar metrics to the run's own
     wandb (resume="must") under a distinct prefix so it doesn't collide with
     the partial in-line eval/* series.

Usage:
  python scripts/adaptive/eval_final_540.py --wid jc264zfr --k 5 --seed 42 \
      --iter 365 --num-rollouts 20 --num-maps 540
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_cmd(ckpt_path, k, sl, num_rollouts, num_maps, num_agents):
    horizon = k * sl
    return [
        sys.executable, "-m", "pufferlib.pufferl", "eval", "puffer_adaptive_drive",
        "--load-model-path", str(ckpt_path),
        "--eval.wosac-realism-eval", "False",
        "--eval.human-replay-eval", "True",
        "--eval.human-replay-num-agents", str(num_agents),
        "--eval.human-replay-num-maps", str(num_maps),
        "--eval.human-replay-num-rollouts", str(num_rollouts),
        "--eval.human-replay-control-mode", "control_vehicles",
        "--eval.map-dir", "resources/drive/binaries/nuplan_hard",
        "--eval.num-maps", str(num_maps),
        "--env.k-scenarios", str(k),
        "--env.scenario-length", str(sl),
        "--train.horizon", str(horizon),
        "--env.goal-behavior", "3",
        "--env.conditioning.type", "none",
    ]


def run_eval(cmd, timeout):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"pufferl eval failed (exit {proc.returncode})")
    out = proc.stdout
    s, e = "HUMAN_REPLAY_METRICS_START", "HUMAN_REPLAY_METRICS_END"
    if s not in out or e not in out:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("no HUMAN_REPLAY_METRICS block in stdout")
    return json.loads(out[out.find(s) + len(s):out.find(e)].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--scenario-length", type=int, default=201)
    ap.add_argument("--num-rollouts", type=int, default=20)
    ap.add_argument("--num-maps", type=int, default=540)
    ap.add_argument("--num-agents", type=int, default=540)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eval540")
    ap.add_argument("--table-prefix", default="eval540_20r")
    ap.add_argument("--wandb-project", default="adaptive_aligned_v2")
    ap.add_argument("--wandb-entity", default="emerge_")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=14400)
    args = ap.parse_args()

    ckpt = REPO_ROOT / "experiments" / f"puffer_adaptive_drive_{args.wid}" / \
        f"model_puffer_adaptive_drive_{args.iter:06d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval540] wid={args.wid} k={args.k} seed={args.seed} iter={args.iter} "
          f"maps={args.num_maps} rollouts={args.num_rollouts} horizon={args.k*args.scenario_length}",
          flush=True)
    cmd = build_cmd(ckpt, args.k, args.scenario_length, args.num_rollouts,
                    args.num_maps, args.num_agents)
    metrics = run_eval(cmd, timeout=args.timeout_sec)
    print(f"[eval540]   {len(metrics)} metric keys", flush=True)

    per_agent_log = metrics.get("per_agent_success_log")
    if not per_agent_log:
        raise RuntimeError("no per_agent_success_log in metrics — cannot build per-map table")

    from pufferlib.utils import _build_per_map_wandb_payload
    payload = _build_per_map_wandb_payload(per_agent_log)
    table = payload.get("eval_maps/per_map_summary")
    if table is None:
        raise RuntimeError("no per_map_summary in payload")

    # local CSV (same naming the plot scripts parse)
    csv_path = args.out_dir / f"per_map_k{args.k}_seed{args.seed}_{args.wid}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(table.columns)
        w.writerows(table.data)
    print(f"[eval540]   wrote {csv_path} ({len(table.data)} maps, cols={table.columns})", flush=True)

    if not args.no_wandb:
        import wandb
        run = wandb.init(id=args.wid, project=args.wandb_project,
                         entity=args.wandb_entity, resume="must")
        wandb.define_metric(f"{args.table_prefix}_step")
        wandb.define_metric(f"{args.table_prefix}/*", step_metric=f"{args.table_prefix}_step")
        log = {f"{args.table_prefix}_step": args.iter}
        for k, v in payload.items():
            log[k.replace("eval_maps/", f"{args.table_prefix}/")] = v
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log[f"{args.table_prefix}/human_replay_{k}"] = v
        run.log(log)
        run.finish()
        print(f"[eval540]   logged {args.table_prefix}/* table to wandb run {args.wid}", flush=True)


if __name__ == "__main__":
    main()
