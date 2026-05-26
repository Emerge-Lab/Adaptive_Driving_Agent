"""Offline human-replay eval for one wid (loops over all 8 checkpoints) from
the k=4 / gb=3 trial-mode sweep, appended to the original wandb run.

Mirrors the in-line eval that `pufferlib.utils.run_human_replay_eval_in_subprocess`
fires during training (same dataset, same eval-set size, same gb=3/k=4 trial
config) so the offline numbers are apples-to-apples with the in-line samples
that landed in wandb before the 24h wallclock killed the runs.

For each iter in {10,20,30,40,50,60,70,76} we:
    1. spawn a `puffer eval` subprocess against the matching checkpoint,
    2. parse the HUMAN_REPLAY_METRICS JSON block, and
    3. log `offline_eval/human_replay_<key>` against a custom step-axis
       `offline_eval_iter` (the checkpoint iter), on the original wandb run
       (resume="must"). Custom axis sidesteps wandb's step-monotonicity
       constraint on resumed runs; `offline_eval/` prefix keeps post-hoc
       numbers from colliding with the partial in-line `eval/*` series.

Usage:
    python scripts/adaptive/eval_k4_gb3_one.py --wid rwg5a65x
    python scripts/adaptive/eval_k4_gb3_one.py --wid rwg5a65x --iters 76
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ITERS = (10, 20, 30, 40, 50, 60, 70, 76)


def build_cmd(ckpt_path: Path, num_rollouts: int, num_maps: int, num_agents: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pufferlib.pufferl",
        "eval",
        "puffer_adaptive_drive",
        "--load-model-path",
        str(ckpt_path),
        "--eval.wosac-realism-eval",
        "False",
        "--eval.human-replay-eval",
        "True",
        "--eval.human-replay-num-agents",
        str(num_agents),
        "--eval.human-replay-num-maps",
        str(num_maps),
        "--eval.human-replay-num-rollouts",
        str(num_rollouts),
        "--eval.human-replay-control-mode",
        "control_vehicles",
        "--eval.map-dir",
        "resources/drive/binaries/nuplan_hard",
        "--eval.num-maps",
        str(num_maps),
        "--env.k-scenarios",
        "4",
        "--env.scenario-length",
        "201",
        "--train.horizon",
        "804",
        "--env.goal-behavior",
        "3",
        "--env.conditioning.type",
        "none",
    ]


def run_eval(cmd: list[str], timeout: int) -> dict:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"puffer eval failed (exit {proc.returncode})")

    out = proc.stdout
    start_tag = "HUMAN_REPLAY_METRICS_START"
    end_tag = "HUMAN_REPLAY_METRICS_END"
    if start_tag not in out or end_tag not in out:
        sys.stderr.write(out)
        raise RuntimeError("no HUMAN_REPLAY_METRICS block in stdout")
    start = out.find(start_tag) + len(start_tag)
    end = out.find(end_tag)
    return json.loads(out[start:end].strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", required=True, help="wandb run id (= experiment dir suffix)")
    ap.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERS),
        help="checkpoint iters to eval (default: 10 20 30 40 50 60 70 76)",
    )
    ap.add_argument("--num-rollouts", type=int, default=20)
    ap.add_argument("--num-maps", type=int, default=540)
    ap.add_argument("--num-agents", type=int, default=540)
    ap.add_argument("--wandb-project", default="adaptive_aligned_v2")
    ap.add_argument("--wandb-entity", default="emerge_")
    ap.add_argument("--no-wandb", action="store_true", help="skip wandb logging (smoke-test)")
    ap.add_argument("--timeout-sec", type=int, default=3600, help="per-iter eval timeout")
    args = ap.parse_args()

    ckpt_dir = REPO_ROOT / "experiments" / f"puffer_adaptive_drive_{args.wid}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(ckpt_dir)

    # Resolve all checkpoint paths up-front so we fail fast if any are missing.
    iter_to_path: dict[int, Path] = {}
    for it in args.iters:
        p = ckpt_dir / f"model_puffer_adaptive_drive_{it:06d}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        iter_to_path[it] = p

    # One wandb session for the whole wid (avoids 8× init/finish overhead).
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            id=args.wid,
            project=args.wandb_project,
            entity=args.wandb_entity,
            resume="must",
        )
        wandb.define_metric("offline_eval_iter")
        wandb.define_metric("offline_eval/*", step_metric="offline_eval_iter")

    try:
        for it in args.iters:
            ckpt_path = iter_to_path[it]
            print(f"[eval_k4_gb3_one] wid={args.wid} iter={it} ckpt={ckpt_path.name}", flush=True)
            cmd = build_cmd(ckpt_path, args.num_rollouts, args.num_maps, args.num_agents)
            metrics = run_eval(cmd, timeout=args.timeout_sec)
            print(f"[eval_k4_gb3_one]   got {len(metrics)} metric keys", flush=True)

            if run is not None:
                log = {"offline_eval_iter": it}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        log[f"offline_eval/human_replay_{k}"] = v
                per_agent_log = metrics.get("per_agent_success_log")
                if per_agent_log:
                    try:
                        from pufferlib.utils import _build_per_map_wandb_payload
                        for k, v in _build_per_map_wandb_payload(per_agent_log).items():
                            log[k.replace("eval_maps/", "offline_eval_maps/")] = v
                    except Exception as e:
                        print(f"[eval_k4_gb3_one]   per-map payload failed: {e}", flush=True)
                run.log(log)
                print(f"[eval_k4_gb3_one]   logged offline_eval/* @ iter={it}", flush=True)
            else:
                print(json.dumps(metrics, indent=2))
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
