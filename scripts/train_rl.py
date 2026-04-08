#!/usr/bin/env python3
"""
Train a single RL policy on the table tennis environment.

Usage
-----
# Task-space, SAC, Numerical IK, MinimumJerk trajectory:
python scripts/train_rl.py \\
    --algorithm SAC \\
    --mode task_space \\
    --ik NumericalIKSolver \\
    --traj MinimumJerkTrajectory

# Joint-space, PPO, no IK:
python scripts/train_rl.py \\
    --algorithm PPO \\
    --mode joint_space \\
    --traj CubicSplineTrajectory \\
    --steps 1000000

# Quick smoke-test (10k steps):
python scripts/train_rl.py --algorithm SAC --steps 10000 --no-eval
"""

import sys, argparse, os, json, time, csv
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.utils import load_config
from src.rl.gym_env import make_env

def parse_args():
    cfg = load_config("config/rl.yaml")
    p = argparse.ArgumentParser(description="Train RL policy for table tennis robot")
    p.add_argument("--algorithm", default="PPO", choices=cfg["algorithms"], help="RL algorithm")
    p.add_argument("--mode", default=cfg["mode"], choices=["task_space", "joint_space"], help="Action space mode")
    p.add_argument("--ik", default="NumericalIKSolver", choices=cfg["ik_solvers"] + ["None"], help="IK solver (task_space only)")
    p.add_argument("--traj", default="MinimumJerkTrajectory", choices=cfg["trajectory_planners"], help="Trajectory planner")
    p.add_argument("--steps", type=int, default=cfg["training"]["total_timesteps"], help="Total training timesteps")
    p.add_argument("--seed", type=int, default=cfg["training"]["seed"])
    p.add_argument("--action-repeat", type=int, default=cfg["env"]["action_repeat"])
    p.add_argument("--log-dir", default=cfg["logging"]["log_dir"])
    p.add_argument("--no-eval", action="store_true", help="Skip periodic evaluation (faster)")
    p.add_argument("--render", action="store_true", help="Launch MuJoCo viewer during evaluation")
    p.add_argument("--load", default=None, help="Path to existing model to continue training")
    return p.parse_args(), cfg

def build_model(algorithm: str, env, cfg: dict, seed: int):
    """Instantiate SB3/sb3-contrib model with config hyper-parameters."""
    hp = cfg["hyperparams"].get(algorithm, {})
    policy = hp.pop("policy", "MlpPolicy")

    import stable_baselines3 as sb3
    import sb3_contrib

    ALG_MAP = {
        "PPO":          sb3.PPO,
        "SAC":          sb3.SAC,
        "TD3":          sb3.TD3,
        "DDPG":         sb3.DDPG,
        "A2C":          sb3.A2C,
        "TQC":          sb3_contrib.TQC,
        "RecurrentPPO": sb3_contrib.RecurrentPPO,
    }

    cls = ALG_MAP[algorithm]
    model = cls(policy, env, verbose=1, seed=seed, **hp)
    # Restore pop'd value so cfg stays intact for next call
    hp["policy"] = policy
    return model


def run_eval(model, env, n_episodes: int) -> dict:
    """
    Evaluate model for n_episodes.

    Returns
    -------
    dict with mean_reward, std_reward, hit_rate, other_side_hit_rate,
    land_rate, net_rate
    """
    rewards, hits, other_side_hits, lands, nets = [], [], [], [], []

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_count  = 0
    ep_hits = ep_other_side = ep_land = ep_net = False

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_hits  |= info.get("hit_detected", False)
        ep_other_side |= info.get("hit_with_other_side", False)
        ep_land  |= info.get("landed_far_side", False)
        ep_net   |= info.get("over_net", False)

        if terminated or truncated:
            rewards.append(ep_reward)
            hits.append(float(ep_hits))
            other_side_hits.append(float(ep_other_side))
            lands.append(float(ep_land))
            nets.append(float(ep_net))
            ep_reward = 0.0
            ep_hits = ep_other_side = ep_land = ep_net = False
            ep_count += 1
            obs, _ = env.reset()

    import numpy as np
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "hit_rate":    float(np.mean(hits)),
        "other_side_hit_rate": float(np.mean(other_side_hits)),
        "land_rate":   float(np.mean(lands)),
        "net_rate":    float(np.mean(nets)),
    }


def main():
    args, cfg = parse_args()

    # ── Build run identifier ────────────────────────────────────────
    ik_tag = args.ik if args.mode == "task_space" else "noIK"
    run_id = f"{args.algorithm}_{args.mode}_{ik_tag}_{args.traj}_seed{args.seed}"
    log_dir = Path(args.log_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  TABLE TENNIS RL TRAINING")
    print("=" * 68)
    print(f"  Run ID    : {run_id}")
    print(f"  Algorithm : {args.algorithm}")
    print(f"  Mode      : {args.mode}")
    print(f"  IK        : {args.ik}")
    print(f"  Trajectory: {args.traj}")
    print(f"  Steps     : {args.steps:,}")
    print(f"  Log dir   : {log_dir}")
    print()

    # ── Save run config ─────────────────────────────────────────────
    run_cfg = vars(args)
    with open(log_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    # ── Build environments ──────────────────────────────────────────
    train_env = make_env(
        mode          = args.mode,
        ik_name       = args.ik,
        traj_name     = args.traj,
        randomize     = True,
        action_repeat = args.action_repeat,
        scene_xml     = cfg["env"]["scene_xml"],
        sim_cfg_path  = cfg["env"]["sim_cfg_path"],
        robot_cfg_path= cfg["env"]["robot_cfg_path"],
    )

    eval_env = make_env(
        mode          = args.mode,
        ik_name       = args.ik,
        traj_name     = args.traj,
        randomize     = True,
        action_repeat = args.action_repeat,
        render_mode   = "human" if args.render else None,
        scene_xml     = cfg["env"]["scene_xml"],
        sim_cfg_path  = cfg["env"]["sim_cfg_path"],
        robot_cfg_path= cfg["env"]["robot_cfg_path"],
    )

    # ── Build or load model ─────────────────────────────────────────
    if args.load:
        print(f"  Loading model from {args.load}")
        from stable_baselines3 import SAC  # fallback; actual class resolved below
        model = build_model(args.algorithm, train_env, cfg, args.seed)
        model.set_parameters(args.load)
    else:
        model = build_model(args.algorithm, train_env, cfg, args.seed)

    # ── CSV metrics log ─────────────────────────────────────────────
    metrics_path = log_dir / "metrics.csv"
    csv_fields = ["timestep", "mean_reward", "std_reward",
                  "hit_rate", "other_side_hit_rate", "land_rate", "net_rate", "wall_time"]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    # ── Training loop with periodic eval ───────────────────────────
    eval_cfg   = cfg["training"]
    eval_freq  = eval_cfg["eval_freq"]
    n_eval_ep  = eval_cfg["eval_episodes"]
    save_freq  = cfg["logging"]["save_freq"]

    steps_done = 0
    t_start    = time.time()

    while steps_done < args.steps:
        chunk = min(eval_freq, args.steps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=(steps_done == 0))
        steps_done += chunk

        # Save checkpoint
        if steps_done % save_freq < eval_freq:
            ckpt_path = log_dir / f"model_{steps_done}"
            model.save(str(ckpt_path))
            print(f"  [step {steps_done:>8,}] checkpoint saved → {ckpt_path}.zip")

        # Evaluate
        if not args.no_eval:
            metrics = run_eval(model, eval_env, n_eval_ep)
            metrics["timestep"] = steps_done
            metrics["wall_time"] = round(time.time() - t_start, 1)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(metrics)
            print(
                f"  [step {steps_done:>8,}]  "
                f"reward={metrics['mean_reward']:+7.2f}±{metrics['std_reward']:.2f}  "
                f"hit={metrics['hit_rate']:.0%}  "
                f"other_side={metrics['other_side_hit_rate']:.0%}  "
                f"net={metrics['net_rate']:.0%}  "
                f"land={metrics['land_rate']:.0%}"
            )

    # ── Save final model ────────────────────────────────────────────
    final_path = log_dir / "model_final"
    model.save(str(final_path))
    print(f"\n  Final model saved → {final_path}.zip")
    print(f"  Metrics log     → {metrics_path}")
    print(f"  Total wall time : {(time.time()-t_start)/60:.1f} min")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
