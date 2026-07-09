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


def build_cmd(ckpt_path, k, sl, num_rollouts, num_maps, num_agents,
              hidden_size=None, partner_id=None, entropy_ub=None,
              demo_trial_0=False, map_dir="resources/drive/binaries/nuplan_hard"):
    horizon = k * sl
    cmd = [
        sys.executable, "-m", "pufferlib.pufferl", "eval", "puffer_adaptive_drive",
        "--load-model-path", str(ckpt_path),
        "--eval.wosac-realism-eval", "False",
        "--eval.human-replay-eval", "True",
        "--eval.human-replay-num-agents", str(num_agents),
        "--eval.human-replay-num-maps", str(num_maps),
        "--eval.human-replay-num-rollouts", str(num_rollouts),
        "--eval.human-replay-control-mode", "control_vehicles",
        "--eval.map-dir", str(map_dir),
        "--eval.num-maps", str(num_maps),
        "--env.k-scenarios", str(k),
        "--env.scenario-length", str(sl),
        "--train.horizon", str(horizon),
        "--env.goal-behavior", "3",
        "--env.conditioning.type", "none",
        # Match the TRAINING reward weights so the per-trial return we log is
        # comparable to what the agent optimized (see final_runs_manifest.md +
        # cluster_coplayer_grid_k234.sh:44-55). Eval default would inherit
        # adaptive.ini sparse values (lane 0, collision/offroad −0.1).
        "--env.reward-vehicle-collision", "-0.5",
        "--env.reward-offroad-collision", "-0.5",
        "--env.reward-lane-align", "0.05",
    ]
    if hidden_size is not None:
        cmd += [
            "--policy.hidden-size", str(hidden_size),
            "--transformer.input-size", str(hidden_size),
            "--transformer.hidden-size", str(hidden_size),
        ]
    if demo_trial_0:
        cmd += ["--env.demo-trial-0", "True"]
    if partner_id is not None:
        # Match training's co-player env exactly (cluster_hiddensize_ablation.sh:
        # partner=2e029h15, e_ub=0.10, collision/offroad_lb=-2, discount=[0.4,1]).
        cmd += [
            "--env.co-player-enabled", "1",
            "--env.co-player-policy.policy-path", f"experiments/puffer_drive_{partner_id}.pt",
            "--env.co-player-policy.architecture", "Transformer",
            "--env.co-player-policy.transformer.horizon", str(sl),
            "--env.co-player-policy.conditioning.type", "all",
            "--env.co-player-policy.conditioning.collision-weight-lb", "-2",
            "--env.co-player-policy.conditioning.collision-weight-ub", "0",
            "--env.co-player-policy.conditioning.offroad-weight-lb", "-2",
            "--env.co-player-policy.conditioning.offroad-weight-ub", "0",
            "--env.co-player-policy.conditioning.entropy-weight-lb", "0",
            "--env.co-player-policy.conditioning.entropy-weight-ub", str(entropy_ub if entropy_ub is not None else 0.10),
            "--env.co-player-policy.conditioning.discount-weight-lb", "0.4",
            "--env.co-player-policy.conditioning.discount-weight-ub", "1",
            "--env.external-co-player-actions", "True",
            "--env.map-rand-per-scenario", "False",
            "--env.entropy-curriculum-enabled", "False",
        ]
    return cmd


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
    ap.add_argument("--hidden-size", type=int, default=None,
                    help="Override policy/transformer hidden_size for non-default checkpoints.")
    ap.add_argument("--partner-id", default=None,
                    help="Co-player wid (e.g. 2e029h15). When set, eval env mirrors training's co-player setup.")
    ap.add_argument("--entropy-ub", type=float, default=None,
                    help="Co-player entropy-weight upper bound (matches training's ENTROPY_UB, e.g. 0.10).")
    ap.add_argument("--demo-trial-0", action="store_true",
                    help="Replay recorded human trajectory for the ego during trial 0. Trials 1..K-1 policy-driven.")
    ap.add_argument("--map-dir", default="resources/drive/binaries/nuplan_hard",
                    help="Eval map directory (default: nuplan_hard).")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "eval540")
    ap.add_argument("--return-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "eval540_return",
                    help="Where to write per-map RETURN CSV (continuous adaptation metric).")
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
                    args.num_maps, args.num_agents,
                    hidden_size=args.hidden_size,
                    partner_id=args.partner_id, entropy_ub=args.entropy_ub,
                    demo_trial_0=args.demo_trial_0, map_dir=args.map_dir)
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

    # ----- per-map RETURN CSV (continuous adaptation metric) -----
    per_agent_return_log = metrics.get("per_agent_return_log")
    if per_agent_return_log:
        from pufferlib.utils import _build_per_map_return_payload
        return_payload = _build_per_map_return_payload(per_agent_return_log)
        return_table = return_payload.get("eval_maps/per_map_return_summary")
        if return_table is not None:
            args.return_dir.mkdir(parents=True, exist_ok=True)
            ret_csv = args.return_dir / f"per_map_R_k{args.k}_seed{args.seed}_{args.wid}.csv"
            with open(ret_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(return_table.columns)
                w.writerows(return_table.data)
            print(f"[eval540]   wrote {ret_csv} "
                  f"({len(return_table.data)} maps, cols={return_table.columns})", flush=True)
    else:
        print("[eval540]   WARNING: no per_agent_return_log in metrics — return CSV skipped",
              flush=True)

    # ----- per-component CSVs (goal / collision / offroad / lane) -----
    # Sourced from the env-side per-trial accumulators (drive.h trial_R_*),
    # snapshotted in the evaluator at each trial_ended_this_step. Same per-map
    # reducer shape as the return CSV: cols = [map_id, t0..t{K-1}, ada_delta].
    def _write_component_csv(log, suffix):
        if not log:
            print(f"[eval540]   WARNING: no per_agent_{suffix}_log — skipped (non-trial eval?)",
                  flush=True)
            return
        import numpy as np
        trial_keys = sorted(
            [k for k in log[0].keys() if k and k[0] in ("t", "s") and k[1:].isdigit()],
            key=lambda c: int(c[1:]),
        )
        if not trial_keys:
            return
        n_agents = max(r["agent"] for r in log) + 1
        n_rollouts = max(r["rollout"] for r in log) + 1
        K = len(trial_keys)
        grid = np.zeros((n_rollouts, n_agents, K), dtype=np.float32)
        for rec in log:
            for ti, tk in enumerate(trial_keys):
                grid[rec["rollout"], rec["agent"], ti] = float(rec.get(tk, 0.0))
        per_map = grid.mean(axis=0)                       # (n_agents, K)
        ada_delta = per_map[:, -1] - per_map[:, 0]
        out_csv = args.return_dir / f"per_map_{suffix}_k{args.k}_seed{args.seed}_{args.wid}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["map_id", *trial_keys, f"ada_delta_{suffix}_last_minus_0"])
            for m in range(n_agents):
                row = [int(m)] + [float(per_map[m, ti]) for ti in range(K)] + [float(ada_delta[m])]
                w.writerow(row)
        print(f"[eval540]   wrote {out_csv} ({n_agents} maps)", flush=True)

    _write_component_csv(metrics.get("per_agent_goal_log"),      "goal")
    _write_component_csv(metrics.get("per_agent_collision_log"), "collision")
    _write_component_csv(metrics.get("per_agent_offroad_log"),   "offroad")
    _write_component_csv(metrics.get("per_agent_lane_log"),      "lane")

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
