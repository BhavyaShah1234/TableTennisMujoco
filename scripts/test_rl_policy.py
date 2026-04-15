#!/usr/bin/env python3
"""
RL Policy Evaluation Test
=========================

Loads a saved Stable Baselines 3 checkpoint and evaluates it on the
table tennis environment, reporting hit rate, timing, landing accuracy,
and paddle position / orientation error at the moment of closest approach.

Usage
-----
# Evaluate a saved checkpoint (auto-detects algorithm from config.json in same dir):
python scripts/test_rl_policy.py --model results/SAC_task_space_NumericalIKSolver_MinimumJerkTrajectory_seed42/model_final

# Specify algorithm explicitly:
python scripts/test_rl_policy.py --model results/.../model_final --algorithm SAC

# Run headless (no viewer):
python scripts/test_rl_policy.py --model results/.../model_final --no-viewer

# More episodes for statistical significance:
python scripts/test_rl_policy.py --model results/.../model_final --episodes 200
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.gym_env import make_env
from src.utils.utils import load_config


# ── Tolerances for pass/fail metrics ────────────────────────────────────────
POS_TOL_STRICT  = 0.05   # m   — tight: paddle within 5 cm of ball at closest
POS_TOL_LOOSE   = 0.10   # m   — loose: paddle within 10 cm (impact zone)
TIME_TOL        = 0.15   # s   — timing deviation from ideal
ORI_TOL_STRICT  = 30.0   # deg — paddle normal vs desired normal (strict)
ORI_TOL_LOOSE   = 60.0   # deg — loose orientation tolerance


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _detect_algorithm(model_path: str) -> str:
    """Try to auto-detect algorithm from config.json next to the checkpoint."""
    model_dir = Path(model_path).parent
    cfg_path  = model_dir / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("algorithm", "SAC")
        except Exception:
            pass
    return "SAC"


def _load_model(algorithm: str, model_path: str, env):
    """Load SB3 / sb3-contrib model by algorithm name."""
    import stable_baselines3 as sb3
    try:
        import sb3_contrib
    except ImportError:
        sb3_contrib = None

    ALG_MAP = {
        "PPO":          sb3.PPO,
        "SAC":          sb3.SAC,
        "TD3":          sb3.TD3,
        "DDPG":         sb3.DDPG,
        "A2C":          sb3.A2C,
    }
    if sb3_contrib is not None:
        ALG_MAP["TQC"]          = sb3_contrib.TQC
        ALG_MAP["RecurrentPPO"] = sb3_contrib.RecurrentPPO

    cls = ALG_MAP.get(algorithm)
    if cls is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALG_MAP)}")

    # Add .zip if missing
    path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not Path(path).exists():
        # Also try without .zip (some SB3 versions save without extension)
        path = model_path
    return cls.load(path, env=env)


# ── Per-episode metrics collector ────────────────────────────────────────────

class EpisodeMetrics:
    """Collect detailed per-episode measurements for one evaluation episode."""

    def __init__(self):
        self.hit_detected     = False
        self.hit_after_bounce = False
        self.over_net         = False
        self.landed_far_side  = False
        self.missed           = False
        self.episode_reward   = 0.0

        # Closest approach tracking
        self.closest_dist     = float("inf")
        self.closest_paddle   = None   # paddle position at closest approach
        self.closest_normal   = None   # paddle normal at closest approach
        self.ball_vel_before  = None   # ball velocity just before closest
        self.ball_vel_after   = None   # ball velocity just after  closest

        # Bounce + timing
        self.bounce_detected  = False
        self.bounce_time      = None
        self.vx_post_bounce   = None

    def update_closest(self, paddle_pos, paddle_normal, ball_pos, ball_vel, dist):
        if dist < self.closest_dist:
            self.closest_dist   = dist
            self.closest_paddle = paddle_pos.copy()
            self.closest_normal = paddle_normal.copy()
            self.ball_vel_before = ball_vel.copy()

    def finalize_vel_after(self, ball_vel):
        """Call ~20 ms after closest step to get post-hit ball velocity."""
        if self.ball_vel_after is None:
            self.ball_vel_after = ball_vel.copy()


def run_one_episode(env, model, step_limit: int = 7000) -> EpisodeMetrics:
    """Run one evaluation episode; return collected metrics."""
    obs, _ = env.reset()
    metrics = EpisodeMetrics()
    closest_step    = None
    prev_ball_vel_z = None
    was_falling     = False

    for step in range(step_limit):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        metrics.episode_reward += float(reward)

        # Read info flags (cumulative within episode)
        metrics.hit_detected     |= bool(info.get("hit_detected",     False))
        metrics.hit_after_bounce |= bool(info.get("hit_after_bounce", False))
        metrics.over_net         |= bool(info.get("over_net",         False))
        metrics.landed_far_side  |= bool(info.get("landed_far_side",  False))
        metrics.missed           |= bool(info.get("missed",           False))

        # Read live simulation state
        ball      = env.get_ball_state()
        ball_pos  = ball["position"]
        ball_vel  = ball["velocity"]
        paddle_p, _ = env.get_end_effector_pose()
        paddle_n  = env.get_paddle_normal()
        dist      = float(np.linalg.norm(paddle_p - ball_pos))

        # Track first table bounce (detect upward velocity near table z)
        if ball_vel[2] < -0.5:
            was_falling = True
        if (not metrics.bounce_detected and was_falling
                and ball_vel[2] > 0.3 and 0.74 < ball_pos[2] < 0.92):
            metrics.bounce_detected = True
            metrics.bounce_time     = env.get_simulation_time()
            metrics.vx_post_bounce  = ball_vel[0]
            was_falling = False

        # Track closest approach between paddle and ball
        metrics.update_closest(paddle_p, paddle_n, ball_pos, ball_vel, dist)
        if closest_step is None and dist == metrics.closest_dist:
            closest_step = step
        # Capture ball velocity ~20 ms after closest step
        if closest_step is not None and step == closest_step + 20:
            metrics.finalize_vel_after(ball_vel)

        if terminated or truncated:
            break

    # If post-hit velocity was never captured (episode ended early)
    if metrics.ball_vel_after is None:
        metrics.ball_vel_after = env.get_ball_state()["velocity"].copy()

    return metrics


# ── Aggregate statistics ──────────────────────────────────────────────────────

def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n//d if d>0 else 0}%)"


def _summarise(all_metrics: list, algorithm: str, model_path: str) -> int:
    """Print aggregate stats; return number of failed checks (0 = all passed)."""
    n = len(all_metrics)
    if n == 0:
        print("  No episodes recorded.")
        return 1

    hits         = sum(1 for m in all_metrics if m.hit_detected)
    hits_bounce  = sum(1 for m in all_metrics if m.hit_after_bounce)
    overs        = sum(1 for m in all_metrics if m.over_net)
    lands        = sum(1 for m in all_metrics if m.landed_far_side)
    misses       = sum(1 for m in all_metrics if m.missed)

    rewards       = [m.episode_reward for m in all_metrics]
    closest_dists = [m.closest_dist  for m in all_metrics if m.closest_dist < float("inf")]

    # Position error at closest approach (how close did the paddle get?)
    strict_pos  = sum(1 for d in closest_dists if d < POS_TOL_STRICT)
    loose_pos   = sum(1 for d in closest_dists if d < POS_TOL_LOOSE)

    # Orientation error: angle between paddle normal and -ball_vel_before at
    # closest approach (desired normal = oppose incoming ball direction)
    ori_errors = []
    for m in all_metrics:
        if m.closest_normal is not None and m.ball_vel_before is not None:
            spd = float(np.linalg.norm(m.ball_vel_before))
            if spd > 0.1:
                desired = -m.ball_vel_before / spd
                ori_errors.append(_angle_deg(m.closest_normal, desired))

    # vx change: measure paddle hit quality
    vx_improvements = []
    for m in all_metrics:
        if (m.vx_post_bounce is not None
                and m.ball_vel_after is not None
                and m.hit_detected):
            # Ball should slow down (vx_after < vx_post_bounce) after being hit
            vx_improvements.append(m.vx_post_bounce - m.ball_vel_after[0])

    def _stat(arr, label, fmt=".3f"):
        if not arr:
            return f"  {label:40s}: n/a"
        mn = np.mean(arr)
        sd = np.std(arr)
        lo = np.min(arr)
        hi = np.max(arr)
        return f"  {label:40s}: mean={mn:{fmt}}  std={sd:{fmt}}  min={lo:{fmt}}  max={hi:{fmt}}"

    print("=" * 68)
    print("  RL POLICY EVALUATION RESULTS")
    print("=" * 68)
    print(f"  Model      : {model_path}")
    print(f"  Algorithm  : {algorithm}")
    print(f"  Episodes   : {n}")
    print()
    print("  ── Event rates ──────────────────────────────────────────────")
    print(f"  Hit detected (any)     : {_pct(hits, n)}")
    print(f"  Hit after bounce       : {_pct(hits_bounce, n)}")
    print(f"  Ball over net          : {_pct(overs, n)}")
    print(f"  Ball landed far side   : {_pct(lands, n)}")
    print(f"  Missed (no contact)    : {_pct(misses, n)}")
    print()
    print("  ── Position accuracy (closest approach) ─────────────────────")
    print(f"  Strict  < {POS_TOL_STRICT:.2f} m     : {_pct(strict_pos, n)}")
    print(f"  Loose   < {POS_TOL_LOOSE:.2f} m     : {_pct(loose_pos, n)}")
    print(_stat(closest_dists, "Closest dist (m)"))
    print()
    print("  ── Orientation accuracy (closest approach) ──────────────────")
    print(f"  Strict  < {ORI_TOL_STRICT:.0f} deg     : {_pct(sum(1 for e in ori_errors if e < ORI_TOL_STRICT), len(ori_errors) if ori_errors else 1)}")
    print(f"  Loose   < {ORI_TOL_LOOSE:.0f} deg     : {_pct(sum(1 for e in ori_errors if e < ORI_TOL_LOOSE), len(ori_errors) if ori_errors else 1)}")
    print(_stat(ori_errors, "Paddle normal error (deg)", ".1f"))
    print()
    print("  ── Hit quality ──────────────────────────────────────────────")
    print(_stat(vx_improvements, "Ball vx reduction after hit (m/s)"))
    print()
    print("  ── Episode reward ───────────────────────────────────────────")
    print(_stat(rewards, "Episode total reward"))
    print()

    # ── Pass/fail summary ────────────────────────────────────────────────────
    MIN_HIT_RATE    = 0.30   # 30% hit-after-bounce rate expected minimum
    MIN_POS_LOOSE   = 0.40   # 40% of episodes reach impact zone
    MIN_OVER_NET    = 0.15   # 15% of episodes ball goes over net

    checks = [
        (hits_bounce / n >= MIN_HIT_RATE,
         f"Hit-after-bounce rate ≥ {MIN_HIT_RATE:.0%}  (got {hits_bounce/n:.0%})"),
        (loose_pos / n >= MIN_POS_LOOSE,
         f"Paddle enters impact zone ≥ {MIN_POS_LOOSE:.0%}  (got {loose_pos/n:.0%})"),
        (overs / n >= MIN_OVER_NET,
         f"Ball over net ≥ {MIN_OVER_NET:.0%}  (got {overs/n:.0%})"),
    ]

    print("  ── Minimum-bar checks ───────────────────────────────────────")
    n_fail = 0
    for ok, desc in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {desc}")
        if not ok:
            n_fail += 1
    print()
    if n_fail == 0:
        print("  ALL MINIMUM-BAR CHECKS PASSED")
    else:
        print(f"  {n_fail} CHECK(S) FAILED — policy needs more training or tuning")
    print("=" * 68)

    return n_fail


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved RL policy on the table tennis env")
    p.add_argument("--model",     required=True,
                   help="Path to SB3 model checkpoint (with or without .zip)")
    p.add_argument("--algorithm", default=None,
                   help="Algorithm class name (auto-detected from config.json if omitted)")
    p.add_argument("--episodes",  type=int, default=50,
                   help="Number of evaluation episodes (default: 50)")
    p.add_argument("--mode",      default="task_space",
                   choices=["task_space", "joint_space"],
                   help="Action-space mode used during training")
    p.add_argument("--ik",        default="NumericalIKSolver",
                   help="IK solver name (task_space only)")
    p.add_argument("--traj",      default="MinimumJerkTrajectory",
                   help="Trajectory planner name")
    p.add_argument("--no-viewer", action="store_true",
                   help="Run headless (no MuJoCo viewer)")
    p.add_argument("--randomize", action="store_true", default=True,
                   help="Randomize ball spawn each episode (default: True)")
    p.add_argument("--seed",      type=int, default=0,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()

    algorithm = args.algorithm or _detect_algorithm(args.model)

    print("=" * 68)
    print("  TABLE TENNIS RL POLICY EVALUATION")
    print("=" * 68)
    print(f"  Model      : {args.model}")
    print(f"  Algorithm  : {algorithm}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Mode       : {args.mode}")
    print(f"  IK         : {args.ik}")
    print(f"  Trajectory : {args.traj}")
    print(f"  Seed       : {args.seed}")
    print()

    rl_cfg = load_config("config/rl.yaml")
    env = make_env(
        mode          = args.mode,
        ik_name       = args.ik if args.mode == "task_space" else None,
        traj_name     = args.traj,
        randomize     = True,
        action_repeat = rl_cfg["env"]["action_repeat"],
        render_mode   = None,   # viewer is launched separately below
        scene_xml     = rl_cfg["env"]["scene_xml"],
        sim_cfg_path  = rl_cfg["env"]["sim_cfg_path"],
        robot_cfg_path= rl_cfg["env"]["robot_cfg_path"],
    )
    env.reset(seed=args.seed)

    print(f"  Loading model ...", end="", flush=True)
    model = _load_model(algorithm, args.model, env)
    print(" done")
    print()

    all_metrics: list = []
    interrupted = False

    try:
        for ep in range(1, args.episodes + 1):
            metrics = run_one_episode(env, model)
            all_metrics.append(metrics)

            hit_str   = "HIT"   if metrics.hit_after_bounce else "miss"
            net_str   = "NET"   if metrics.over_net         else "    "
            land_str  = "LAND"  if metrics.landed_far_side  else "    "
            print(f"  ep {ep:>4d}/{args.episodes}  "
                  f"closest={metrics.closest_dist:.3f}m  "
                  f"reward={metrics.episode_reward:+7.2f}  "
                  f"{hit_str}  {net_str}  {land_str}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n  Interrupted by user — summarising collected episodes.")
    finally:
        env.close()

    print()
    n_fail = _summarise(all_metrics, algorithm, args.model)

    if interrupted:
        sys.exit(0)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
