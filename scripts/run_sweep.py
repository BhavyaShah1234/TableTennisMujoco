#!/usr/bin/env python3
"""
Sweep over all algorithm × IK × trajectory × mode combinations.

Generates and (optionally) runs up to 60 training runs.

Task-space runs  : 5 algorithms × 2 IK solvers × 4 trajectory planners = 40
Joint-space runs : 5 algorithms × 1 (no IK)   × 4 trajectory planners = 20
                                                              TOTAL = 60

Usage
-----
# Print all 60 configs without running anything:
python scripts/run_sweep.py --dry-run

# Run all combos sequentially with 500k steps each:
python scripts/run_sweep.py --steps 500000

# Run only task_space combos, SAC only:
python scripts/run_sweep.py --modes task_space --algorithms SAC

# Run all combos for joint_space, limited to 2 specific trajectories:
python scripts/run_sweep.py --modes joint_space \\
    --trajs MinimumJerkTrajectory CubicSplineTrajectory

# Resume: skip runs whose model_final.zip already exists:
python scripts/run_sweep.py --skip-done

# Launch N parallel workers via multiprocessing:
python scripts/run_sweep.py --workers 4
"""

import sys, argparse, json, csv, time, subprocess
from pathlib import Path
from itertools import product

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.utils import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_run_configs(cfg: dict,
                      modes=None,
                      algorithms=None,
                      ik_solvers=None,
                      trajectories=None) -> list[dict]:
    """
    Generate the list of run-configuration dicts covering all combinations.
    """
    all_algorithms   = cfg["algorithms"]
    all_ik_solvers   = cfg["ik_solvers"]        # e.g. [NumericalIKSolver, JacobianIKSolver]
    all_trajectories = cfg["trajectory_planners"]
    all_modes        = ["task_space", "joint_space"]

    # Apply user filters
    algorithms   = algorithms   or all_algorithms
    ik_solvers   = ik_solvers   or all_ik_solvers
    trajectories = trajectories or all_trajectories
    modes        = modes        or all_modes

    runs = []

    for mode, algorithm, traj in product(modes, algorithms, trajectories):
        if mode == "task_space":
            for ik in ik_solvers:
                ik_tag = ik
                run_id = f"{algorithm}_{mode}_{ik_tag}_{traj}_seed{cfg['training']['seed']}"
                runs.append({
                    "run_id":    run_id,
                    "mode":      mode,
                    "algorithm": algorithm,
                    "ik":        ik,
                    "traj":      traj,
                })
        else:  # joint_space — no IK
            run_id = f"{algorithm}_{mode}_noIK_{traj}_seed{cfg['training']['seed']}"
            runs.append({
                "run_id":    run_id,
                "mode":      mode,
                "algorithm": algorithm,
                "ik":        "None",
                "traj":      traj,
            })

    return runs


def run_is_done(run_cfg: dict, log_dir: Path) -> bool:
    """Return True if final model already exists for this run."""
    return (log_dir / run_cfg["run_id"] / "model_final.zip").exists()


def print_summary_table(runs: list[dict]):
    """Pretty-print all run configs."""
    header = "{:<6}  {:<15}  {:<12}  {:<20}  {:<35}  {}"
    row    = "{:<6}  {:<15}  {:<12}  {:<20}  {:<35}  {}"
    sep    = "-" * 110
    print(sep)
    print(header.format("#", "Algorithm", "Mode", "IK Solver", "Trajectory", "Run ID"))
    print(sep)
    for i, r in enumerate(runs, 1):
        print(row.format(i, r["algorithm"], r["mode"], r["ik"], r["traj"], r["run_id"]))
    print(sep)
    print(f"  Total runs: {len(runs)}")


def train_single(run_cfg: dict, args, cfg: dict) -> dict:
    """
    Execute one training run via subprocess, forwarding to train_rl.py.
    Returns a result dict with success/failure and timing.
    """
    train_script = Path(__file__).parent / "train_rl.py"
    cmd = [
        sys.executable, str(train_script),
        "--algorithm",    run_cfg["algorithm"],
        "--mode",         run_cfg["mode"],
        "--ik",           run_cfg["ik"],
        "--traj",         run_cfg["traj"],
        "--steps",        str(args.steps),
        "--seed",         str(cfg["training"]["seed"]),
        "--log-dir",      args.log_dir,
        "--action-repeat",str(cfg["env"]["action_repeat"]),
    ]
    if args.no_eval:
        cmd.append("--no-eval")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    return {
        "run_id":    run_cfg["run_id"],
        "algorithm": run_cfg["algorithm"],
        "mode":      run_cfg["mode"],
        "ik":        run_cfg["ik"],
        "traj":      run_cfg["traj"],
        "success":   result.returncode == 0,
        "wall_min":  round(elapsed / 60, 2),
    }


def train_worker(args_tuple):
    """Pool worker: unpack args and call train_single."""
    run_cfg, script_args, cfg = args_tuple
    return train_single(run_cfg, script_args, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(cfg: dict):
    p = argparse.ArgumentParser(description="Sweep RL training over all combinations")

    p.add_argument("--modes",      nargs="+",
                   default=cfg.get("sweep_modes", ["task_space", "joint_space"]),
                   choices=["task_space", "joint_space"],
                   help="Which action-space modes to include")
    p.add_argument("--algorithms", nargs="+",
                   default=None,
                   choices=cfg["algorithms"],
                   help="Subset of algorithms (default: all)")
    p.add_argument("--iks",        nargs="+",
                   default=None,
                   choices=cfg["ik_solvers"],
                   help="Subset of IK solvers for task_space runs (default: all)")
    p.add_argument("--trajs",      nargs="+",
                   default=None,
                   choices=cfg["trajectory_planners"],
                   help="Subset of trajectory planners (default: all)")
    p.add_argument("--steps",      type=int,
                   default=cfg["training"]["total_timesteps"])
    p.add_argument("--log-dir",    default=cfg["logging"]["log_dir"])
    p.add_argument("--dry-run",    action="store_true",
                   help="Print all configs but do not train")
    p.add_argument("--skip-done",  action="store_true",
                   help="Skip runs whose final model already exists")
    p.add_argument("--no-eval",    action="store_true",
                   help="Disable eval callbacks (faster, no metrics)")
    p.add_argument("--workers",    type=int, default=1,
                   help="Parallel workers (default: 1 = sequential)")
    return p.parse_args()


def main():
    cfg  = load_config("config/rl.yaml")
    args = parse_args(cfg)

    # ── Build run list ──────────────────────────────────────────────
    all_runs = build_run_configs(
        cfg,
        modes       = args.modes,
        algorithms  = args.algorithms,
        ik_solvers  = args.iks,
        trajectories= args.trajs,
    )

    log_dir = Path(args.log_dir)

    # ── Optionally skip completed runs ──────────────────────────────
    if args.skip_done:
        todo = [r for r in all_runs if not run_is_done(r, log_dir)]
        skipped = len(all_runs) - len(todo)
        if skipped:
            print(f"  Skipping {skipped} already-completed run(s).")
    else:
        todo = all_runs

    # ── Print table ─────────────────────────────────────────────────
    print_summary_table(todo)

    if args.dry_run or not todo:
        print("  [--dry-run] No training started.")
        return

    # ── Save sweep config ───────────────────────────────────────────
    log_dir.mkdir(parents=True, exist_ok=True)
    sweep_cfg_path = log_dir / "sweep_config.json"
    with open(sweep_cfg_path, "w") as f:
        json.dump({
            "total_runs": len(all_runs),
            "todo_runs":  len(todo),
            "steps":      args.steps,
            "runs":       todo,
        }, f, indent=2)
    print(f"\n  Sweep config saved → {sweep_cfg_path}")

    # ── Open sweep results CSV ──────────────────────────────────────
    results_path = log_dir / "sweep_results.csv"
    csv_fields   = ["run_id", "algorithm", "mode", "ik", "traj", "success", "wall_min"]
    with open(results_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Execute runs ────────────────────────────────────────────────
    t_sweep_start = time.time()
    results       = []

    if args.workers > 1:
        import multiprocessing as mp
        pool_args = [(r, args, cfg) for r in todo]
        with mp.Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(train_worker, pool_args), 1):
                results.append(result)
                _log_result(result, i, len(todo), results_path, csv_fields)
    else:
        for i, run_cfg in enumerate(todo, 1):
            print(f"\n{'='*68}")
            print(f"  RUN {i}/{len(todo)}: {run_cfg['run_id']}")
            print(f"{'='*68}")
            result = train_single(run_cfg, args, cfg)
            results.append(result)
            _log_result(result, i, len(todo), results_path, csv_fields)

    # ── Final summary ───────────────────────────────────────────────
    n_ok   = sum(r["success"] for r in results)
    n_fail = len(results) - n_ok
    total_min = (time.time() - t_sweep_start) / 60

    print("\n" + "=" * 68)
    print("  SWEEP COMPLETE")
    print("=" * 68)
    print(f"  Total runs  : {len(results)}")
    print(f"  Successful  : {n_ok}")
    print(f"  Failed      : {n_fail}")
    print(f"  Wall time   : {total_min:.1f} min")
    print(f"  Results CSV : {results_path}")

    if n_fail:
        print("\n  FAILED RUNS:")
        for r in results:
            if not r["success"]:
                print(f"    ✗ {r['run_id']}")


def _log_result(result: dict, idx: int, total: int,
                results_path: Path, fields: list):
    """Append one result row to the CSV and print a status line."""
    status = "✓" if result["success"] else "✗"
    print(
        f"\n  [{idx}/{total}] {status} {result['run_id']}"
        f"  ({result['wall_min']:.1f} min)"
    )
    with open(results_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(result)


if __name__ == "__main__":
    main()
