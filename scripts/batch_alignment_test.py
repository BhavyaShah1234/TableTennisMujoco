#!/usr/bin/env python3
"""
Batch Paddle Alignment Test
============================

Sweeps a grid of impact positions and normal directions to stress-test
IK + trajectory + simulation stability across the reachable workspace.

Prints a compact summary table — no viewer, headless only.

Usage:
  python scripts/batch_alignment_test.py
  python scripts/batch_alignment_test.py --t-arrive 5.0
"""

import argparse
import sys
import numpy as np
import mujoco
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.gym_env import Environment
from src.kinematics.inverse_kinematics import NumericalIKSolver
from src.planner.trajectory_planner import MinimumJerkTrajectory
from src.utils.utils import normalize_vector


# ── Test matrix ──────────────────────────────────────────────────────────────

# (tag, impact_xyz, normal_xyz)
TEST_CASES = [
    # ── impact [1.0, 0.0, 1.1]  (reference position from prior sessions) ──
    ("ref  [0, 1, 0]",       [1.0,  0.0, 1.1], [ 0.0,  1.0,  0.0]),
    ("ref  [0,-1, 0]",       [1.0,  0.0, 1.1], [ 0.0, -1.0,  0.0]),
    ("ref  [-1,1,0]",        [1.0,  0.0, 1.1], [-1.0,  1.0,  0.0]),
    ("ref  [-1,1,1]",        [1.0,  0.0, 1.1], [-1.0,  1.0,  1.0]),
    ("ref  [-1,-1,-1]",      [1.0,  0.0, 1.1], [-1.0, -1.0, -1.0]),
    ("ref  [-1,1,-1]",       [1.0,  0.0, 1.1], [-1.0,  1.0, -1.0]),

    # ── diverse impact positions: forward reach ──
    ("fwd  [0.8,0,1.0]",     [0.8,  0.0, 1.0], [-1.0,  0.0,  0.0]),
    ("fwd  [0.8,0,1.0]",     [0.8,  0.0, 1.0], [ 0.0,  1.0,  0.0]),
    ("fwd  [0.9,0,0.9]",     [0.9,  0.0, 0.9], [-1.0,  0.0,  0.5]),
    ("fwd  [1.1,0,1.2]",     [1.1,  0.0, 1.2], [-1.0,  0.0,  0.0]),
    ("fwd  [1.2,0,1.0]",     [1.2,  0.0, 1.0], [-1.0,  0.0,  0.0]),

    # ── diverse impact positions: lateral reach ──
    ("lat  [0.9,0.3,1.1]",   [0.9,  0.3, 1.1], [-1.0,  1.0,  0.0]),
    ("lat  [0.9,0.3,1.1]",   [0.9,  0.3, 1.1], [-1.0,  0.0,  0.5]),
    ("lat  [1.0,-0.3,1.1]",  [1.0, -0.3, 1.1], [-1.0, -1.0,  0.0]),
    ("lat  [0.9,-0.2,1.0]",  [0.9, -0.2, 1.0], [-1.0,  0.0,  0.0]),

    # ── diverse impact positions: height variation ──
    ("ht   [0.9,0,0.95]",    [0.9,  0.0, 0.95], [-1.0,  0.5,  0.0]),
    ("ht   [0.9,0,0.95]",    [0.9,  0.0, 0.95], [ 0.0,  1.0,  0.0]),
    ("ht   [0.9,0,1.3]",     [0.9,  0.0, 1.3], [-1.0,  0.0,  0.0]),
    ("ht   [1.0,0,0.9]",     [1.0,  0.0, 0.9], [-0.5,  1.0, -0.5]),
    ("ht   [1.0,0,1.3]",     [1.0,  0.0, 1.3], [-1.0,  0.5,  0.0]),

    # ── random-ish normals ──
    ("mix  [-0.5,0.5,0.7]",  [1.0,  0.0, 1.1], [-0.5,  0.5,  0.7]),
    ("mix  [-0.7,0.3,-0.6]", [0.9,  0.2, 1.0], [-0.7,  0.3, -0.6]),
    ("mix  [-0.3,-0.8,0.5]", [1.0, -0.2, 1.0], [-0.3, -0.8,  0.5]),
]

PASS_POS_M   = 0.005   # 5 mm
PASS_ANG_DEG = 2.0     # 2 degrees
PASS_QDOT7   = 5.0     # rad/s  — anything above this is instability
V_LIM_RAD_S  = 2.0     # joint velocity auto-scale cap


def _angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _quat_to_z_axis(quat):
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)[:, 2]


def run_case(env, ik, traj_gen, impact, desired, t_arrive_base):
    """Run a single alignment case; return (pos_err, ang_err, max_qdot7, t_arrive, note)."""

    env.reset(options={"randomize": False})
    env._park_ball()
    home_pos = env.home_position.copy()
    env.set_robot_joints(home_pos, np.zeros(env.n_dof))
    mujoco.mj_forward(env.model, env.data)

    q_goal, ik_ok = ik.solve(
        target_position=impact,
        target_normal=desired,
        initial_guess=home_pos,
    )

    if not ik_ok:
        return None  # IK failed → unreachable

    # Auto-scale t_arrive so joint velocities stay within limit
    delta_q = np.abs(q_goal - home_pos)
    t_vel = float(np.max(delta_q * 1.875 / V_LIM_RAD_S)) if np.any(delta_q > 1e-6) else 0.0
    t_arrive = max(t_arrive_base, t_vel)

    traj = traj_gen.generate_trajectory(
        np.array([home_pos, q_goal]),
        np.array([0.0, t_arrive]),
        dt=env.dt,
    )

    total_steps = int((t_arrive + 2.0) / env.dt) + 1
    max_qdot7 = 0.0

    for step in range(total_steps):
        ctrl = traj[min(step, len(traj) - 1)]["position"]
        env._sim_step(ctrl)
        qdot7 = abs(float(env.data.qvel[6]))
        if qdot7 > max_qdot7:
            max_qdot7 = qdot7

    # Final metrics
    paddle_pos, _ = env.get_end_effector_pose()
    paddle_normal = env.get_paddle_normal()
    pos_err = float(np.linalg.norm(paddle_pos - impact))
    ang_err = _angle_deg(paddle_normal, desired)

    q_final = env.get_robot_state()["position"]
    q_track_err = float(np.linalg.norm(q_final - q_goal))
    # If joint tracking error is large, robot couldn't physically reach q_goal
    if q_track_err > 0.05:
        return None  # physically unreachable despite IK success

    return pos_err, ang_err, max_qdot7, t_arrive


def main():
    parser = argparse.ArgumentParser(description="Batch paddle alignment test")
    parser.add_argument("--t-arrive", type=float, default=5.0,
                        help="Base arrival time (auto-scaled up if needed)")
    args = parser.parse_args()

    env = Environment(scene_xml="assets/scene.xml", randomize=False)
    ik = NumericalIKSolver(
        model=env.model,
        data=env.data,
        end_effector_body="paddle",
        end_effector_site="paddle_contact",
        end_effector_normal_site="paddle_normal",
        position_weight=1.0,
        orientation_weight=0.8,
        max_iterations=500,
    )
    traj_gen = MinimumJerkTrajectory()

    header = f"{'Case':<28} {'impact':>14} {'normal':>14} {'t_arr':>5}  {'pos_err':>8} {'ang_err':>8} {'qdot7':>7}  {'status'}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    n_pass = n_fail = n_unreachable = 0

    for tag, impact_raw, normal_raw in TEST_CASES:
        impact  = np.array(impact_raw,  dtype=np.float64)
        desired = normalize_vector(np.array(normal_raw, dtype=np.float64))

        result = run_case(env, ik, traj_gen, impact, desired, args.t_arrive)

        imp_str = f"[{impact[0]:.1f},{impact[1]:.1f},{impact[2]:.1f}]"
        nor_str = f"[{normal_raw[0]:.1f},{normal_raw[1]:.1f},{normal_raw[2]:.1f}]"

        if result is None:
            print(f"{tag:<28} {imp_str:>14} {nor_str:>14}  {'--':>5}  {'--':>8} {'--':>8} {'--':>7}  UNREACHABLE")
            n_unreachable += 1
            continue

        pos_err, ang_err, max_qdot7, t_arrive = result
        ok = (pos_err < PASS_POS_M) and (ang_err < PASS_ANG_DEG) and (max_qdot7 < PASS_QDOT7)
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1

        print(f"{tag:<28} {imp_str:>14} {nor_str:>14} {t_arrive:>5.1f}  "
              f"{pos_err:>7.4f}m {ang_err:>7.2f}°  {max_qdot7:>6.4f}  {status}")

    print(sep)
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  UNREACHABLE: {n_unreachable}")
    print(sep + "\n")

    env.close()


if __name__ == "__main__":
    main()
