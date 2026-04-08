#!/usr/bin/env python3
"""
Robot Speed Benchmark
======================

Answers: "Given a target impact position, what is the minimum time in which
the paddle can travel from the home (ready) position to that point?"

For a grid of target positions across the workspace the script:
  1. Solves IK for each target.
  2. Runs a forward simulation with a min-jerk joint trajectory for a
     series of candidate arrival times (T_list).
  3. Reports the actual paddle-to-target position error at t = T.
  4. Marks the target as REACHABLE within POS_TOL = 0.08 m if the
     error is small enough.
  5. Records the minimum T that satisfies the tolerance → min_reach_time.

Results are printed as a table and a compact heat-map style ASCII grid.

Usage
-----
  python scripts/test_robot_speed.py
  python scripts/test_robot_speed.py --no-viewer   # skip viewer (faster)
"""

import sys, time
import numpy as np
from pathlib import Path
from itertools import product

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.gym_env import Environment
from src.kinematics.inverse_kinematics import NumericalIKSolver
from src.planner.trajectory_planner import MinimumJerkTrajectory
from src.utils.utils import load_config

# ──────────────────────────────────────────────────────────────────
# BENCHMARK PARAMETERS
# ──────────────────────────────────────────────────────────────────

# Target positions to test  (x, y, z) in world frame
# Spread across the robot's reachable workspace from multiple angles
TARGET_POSITIONS = [
    # Close / centre
    (1.10,  0.00, 1.10),
    (1.10,  0.00, 1.35),
    (1.10,  0.00, 0.90),
    # Forward reach (far from base in x)
    (0.85,  0.00, 1.00),
    (0.85,  0.00, 1.30),
    (0.85,  0.00, 0.85),
    # Far lateral (high y)
    (1.00,  0.50, 1.10),
    (1.00, -0.50, 1.10),
    (1.00,  0.60, 1.00),
    (1.00, -0.60, 1.00),
    # High targets
    (1.00,  0.00, 1.50),
    (1.00,  0.25, 1.60),
    (1.10,  0.00, 1.65),
    # Low targets
    (1.00,  0.00, 0.82),
    (0.90,  0.20, 0.82),
    # Long reach (far x, off-centre y)
    (0.82,  0.30, 1.10),
    (0.82, -0.30, 1.10),
    (0.82,  0.50, 1.00),
    # Near-base (x ≈ 1.4)
    (1.35,  0.00, 1.00),
    (1.35,  0.20, 1.10),
    (1.30, -0.30, 1.20),
]

# Arrival times to probe (seconds)
T_LIST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]

# Tolerance: arm must arrive within this distance of target (m)
POS_TOL = 0.08

# IK initial guesses (multi-start to avoid local minima)
IK_GUESSES = [
    np.array([ 0.0, -0.5,  0.0, -1.8,  0.0, 1.5,  0.785]),  # ready (home)
    np.array([ 0.0, -0.3,  0.0, -1.5,  0.0, 1.2,  0.785]),  # mid-reach
    np.array([ 0.0,  0.2,  0.0, -2.2,  0.0, 1.8,  0.785]),  # low
    np.array([ 0.3, -0.4,  0.0, -1.6,  0.3, 1.4,  0.785]),  # lateral A
    np.array([-0.3, -0.4,  0.0, -1.6, -0.3, 1.4,  0.785]),  # lateral B
]


def solve_ik_best(ik, target_pos, model, data):
    """Try multiple IK seeds and return the joint config with lowest FK error."""
    import mujoco
    best_q, best_err = None, float("inf")
    for guess in IK_GUESSES:
        q, _ = ik.solve(target_position=target_pos,
                target_orientation=None,
                initial_guess=guess)
        # FK check
        prev = data.qpos[:7].copy()
        data.qpos[:7] = q
        mujoco.mj_fwdPosition(model, data)
        fk_pos = data.xpos[model.body("paddle").id].copy()
        err = float(np.linalg.norm(fk_pos - target_pos))
        data.qpos[:7] = prev
        mujoco.mj_fwdPosition(model, data)
        if err < best_err:
            best_err = err
            best_q   = q.copy()
    return best_q, best_err


def run_reach_test(env, ik, traj_gen, home_pos, target_pos, t_arrive):
    """
    Simulate the arm moving from home_pos to the IK goal in t_arrive seconds.
    Returns (paddle_pos_at_t, error_at_t, q_goal, ik_fk_err).
    """
    env.reset()
    env.set_robot_joints(home_pos, np.zeros(7))
    import mujoco
    mujoco.mj_fwdPosition(env.model, env.data)

    q_goal, ik_fk_err = solve_ik_best(ik, target_pos, env.model, env.data)

    traj = traj_gen.generate_trajectory(
        np.array([home_pos, q_goal]),
        np.array([0.0, t_arrive]),
        dt=env.dt,
    )

    ctrl = home_pos.copy()
    paddle_at_t = None

    n_steps = int(t_arrive / env.dt) + 20   # a bit past arrival time
    for step in range(n_steps):
        t = env.get_simulation_time()

        if step < len(traj):
            ctrl = traj[step]["position"]
        else:
            ctrl = traj[-1]["position"]

        env._sim_step(ctrl)

        if t >= t_arrive and paddle_at_t is None:
            paddle_at_t, _ = env.get_end_effector_pose()

    if paddle_at_t is None:
        paddle_at_t, _ = env.get_end_effector_pose()

    err = float(np.linalg.norm(paddle_at_t - target_pos))
    return paddle_at_t, err, q_goal, ik_fk_err


def main():
    print("=" * 72)
    print("  TABLE TENNIS ROBOT – SPEED / REACH BENCHMARK")
    print("=" * 72)
    print(f"  Targets   : {len(TARGET_POSITIONS)}")
    print(f"  Times     : {T_LIST}")
    print(f"  Tolerance : {POS_TOL:.3f} m")
    print()

    sim_cfg   = load_config("config/simulation.yaml")
    robot_cfg = load_config("config/robot.yaml")
    env       = Environment(scene_xml="assets/scene.xml", randomize=False)

    ik = NumericalIKSolver(
        model=env.model, data=env.data,
        end_effector_body="paddle",
        position_weight=1.0, orientation_weight=0.0,
        max_iterations=600,
    )
    traj_gen  = MinimumJerkTrajectory()
    home_pos = np.array(robot_cfg["robot"]["home_position"])

    # Pre-compute IK FK errors for each target (headless, once)
    print("  Solving IK for all targets ...", flush=True)
    ik_errors = {}
    for tp in TARGET_POSITIONS:
        env.reset()
        env.set_robot_joints(home_pos, np.zeros(7))
        import mujoco
        mujoco.mj_fwdPosition(env.model, env.data)
        _, err = solve_ik_best(ik, np.array(tp), env.model, env.data)
        ik_errors[tp] = err
    print("  done.\n")

    # Column header
    time_hdr = "  ".join(f"{t:5.2f}" for t in T_LIST)
    col_w    = 7 * len(T_LIST) + 10
    hdr_tgt  = f"  {'Target (x,y,z)':<28s}  IK_err  "
    hdr_T    = "  ".join(f"T={t:.2f}" for t in T_LIST)
    SEP      = "-" * (len(hdr_tgt) + len(hdr_T))

    print(SEP)
    print(hdr_tgt + hdr_T)
    print(SEP)

    results = {}   # target → {T → err}

    for tp in TARGET_POSITIONS:
        tp_arr    = np.array(tp)
        ik_err    = ik_errors[tp]
        row_errs  = {}
        row_str   = f"  ({tp[0]:.2f},{tp[1]:+.2f},{tp[2]:.2f})   {ik_err:6.4f}   "

        for T in T_LIST:
            _, sim_err, _, _ = run_reach_test(env, ik, traj_gen, home_pos, tp_arr, T)
            row_errs[T] = sim_err
            ok = "✓" if sim_err < POS_TOL else "✗"
            row_str += f" {sim_err:.4f}{ok} "

        results[tp] = row_errs
        print(row_str)

    print(SEP)

    # ── Summary: min reach time per target ─────────────────────────────────
    print("\n" + "=" * 72)
    print("  MINIMUM REACH TIME  (first T where sim error < {:.3f} m)".format(POS_TOL))
    print("=" * 72)
    print(f"  {'Target (x,y,z)':<28s}  IK_fk_err  min_T    note")
    print("-" * 72)

    reachable   = []
    unreachable = []

    for tp in TARGET_POSITIONS:
        ik_err   = ik_errors[tp]
        row_errs = results[tp]

        min_T = None
        for T in T_LIST:
            if row_errs[T] < POS_TOL:
                min_T = T
                break

        note = ""
        if ik_err > 0.12:
            note = "⚠ IK error large – target may be out of reach"
        elif ik_err > 0.06:
            note = "IK approx"

        if min_T is not None:
            print(f"  ({tp[0]:.2f},{tp[1]:+.2f},{tp[2]:.2f})   {ik_err:.4f}     {min_T:.2f}s   {note}")
            reachable.append((tp, min_T))
        else:
            best_T   = min(T_LIST, key=lambda T: row_errs[T])
            best_err = row_errs[best_T]
            print(f"  ({tp[0]:.2f},{tp[1]:+.2f},{tp[2]:.2f})   {ik_err:.4f}      —      "
                  f"NOT reached  (best {best_err:.4f}m @ T={best_T:.2f}s)  {note}")
            unreachable.append(tp)

    print("-" * 72)
    print(f"\n  Reachable within {POS_TOL:.3f} m : {len(reachable)}/{len(TARGET_POSITIONS)} targets")
    if reachable:
        min_Ts = [t for _, t in reachable]
        print(f"  Fastest  reach             : {min(min_Ts):.2f} s")
        print(f"  Median   reach             : {np.median(min_Ts):.2f} s")
        print(f"  Slowest  reach             : {max(min_Ts):.2f} s")

    env.close()

    # ── ASCII heat-map ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MIN REACH TIME MAP  (✓ = reachable, value = T_min in s;  — = not reachable)")
    print("  Rows = z (high→low),  Cols = x (close→far);  each cell: y≈0")
    print("=" * 72)

    xs = sorted({tp[0] for tp in TARGET_POSITIONS})
    zs = sorted({tp[2] for tp in TARGET_POSITIONS}, reverse=True)

    hdr = "  z\\x  " + "  ".join(f"{x:.2f}" for x in xs)
    print(hdr)
    for z in zs:
        row = f"  {z:.2f}  "
        for x in xs:
            # Find nearest y≈0 entry
            matches = [tp for tp in TARGET_POSITIONS
                       if tp[0] == x and tp[2] == z and abs(tp[1]) < 0.05]
            if matches:
                tp   = matches[0]
                min_T = next((T for T in T_LIST if results[tp][T] < POS_TOL), None)
                row += (f" {min_T:.2f}✓ " if min_T else "  —   ")
            else:
                row += "      "
        print(row)


if __name__ == "__main__":
    main()
