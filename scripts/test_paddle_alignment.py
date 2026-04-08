#!/usr/bin/env python3
"""
Paddle Alignment Test
=====================

Move the robot arm to a specified impact position and align the paddle normal
with a desired direction vector (tail at impact position). This is useful to
verify that IK uses the paddle contact site (not just the last joint link).

Usage examples:
  python scripts/test_paddle_alignment.py \
	--impact 1.0 0.0 1.0 \
	--normal 1.0 0.0 0.0

  python scripts/test_paddle_alignment.py \
	--impact 0.9 0.2 1.1 \
	--normal 0.0 1.0 0.0 \
	--t-arrive 0.6

  python scripts/test_paddle_alignment.py --no-viewer
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.gym_env import Environment
from src.kinematics.inverse_kinematics import NumericalIKSolver
from src.planner.trajectory_planner import MinimumJerkTrajectory
from src.utils.utils import normalize_vector, load_config


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
	na = float(np.linalg.norm(a))
	nb = float(np.linalg.norm(b))
	if na < 1e-9 or nb < 1e-9:
		return 0.0
	return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
	mat = np.zeros(9, dtype=np.float64)
	mujoco.mju_quat2Mat(mat, quat)
	return mat.reshape(3, 3)[:, 2]


def _sync_viewer_safe(viewer, dt: float) -> bool:
	if viewer is None:
		return False
	try:
		is_running = getattr(viewer, "is_running", None)
		if callable(is_running) and not is_running():
			return False
		viewer.sync()
		if dt > 0.0:
			time.sleep(dt)
		return True
	except Exception:
		return False


def _update_debug_geoms(viewer, impact_pos: np.ndarray, desired_dir: np.ndarray,
						paddle_pos: np.ndarray, paddle_dir: np.ndarray, scale: float = 0.2) -> None:
	if viewer is None:
		return

	user_scn = getattr(viewer, "user_scn", None)
	if user_scn is None:
		return

	user_scn.ngeom = 0

	def _add_arrow(start, color, direction):
		if user_scn.ngeom >= user_scn.maxgeom:
			return
		geom = user_scn.geoms[user_scn.ngeom]
		user_scn.ngeom += 1
		end = start + direction * scale
		mujoco.mjv_connector(
			geom,
			mujoco.mjtGeom.mjGEOM_ARROW,
			0.01,
			start,
			end,
		)
		geom.rgba[:] = np.array(color, dtype=np.float32)

	def _add_sphere(color, radius=0.02):
		if user_scn.ngeom >= user_scn.maxgeom:
			return
		geom = user_scn.geoms[user_scn.ngeom]
		user_scn.ngeom += 1
		mujoco.mjv_initGeom(
			geom,
			mujoco.mjtGeom.mjGEOM_SPHERE,
			np.array([radius, 0.0, 0.0]),
			impact_pos,
			np.eye(3).reshape(9),
			np.array(color, dtype=np.float32),
		)

	# Impact position (yellow sphere)
	_add_sphere([1.0, 1.0, 0.0, 0.9])
	# Desired orientation (green arrow at impact point)
	_add_arrow(impact_pos, [0.0, 1.0, 0.0, 0.9], desired_dir)
	# Current paddle normal (blue arrow at paddle contact point)
	_add_arrow(paddle_pos, [0.1, 0.2, 1.0, 0.9], paddle_dir)


def _apply_position_gains(env: Environment, kp: np.ndarray, kd: np.ndarray) -> None:
	# Map PD gains onto the first n_dof position actuators.
	for i in range(env.n_dof):
		env.model.actuator_gainprm[i, 0] = kp[i]
		env.model.actuator_biasprm[i, 1] = -kp[i]
		env.model.actuator_biasprm[i, 2] = -kd[i]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Paddle alignment IK test")
	parser.add_argument("--impact", nargs=3, type=float, default=[1.0, 0.0, 1.1],
						help="Impact position (x y z) in world frame")
	parser.add_argument("--normal", nargs=3, type=float, default=[-1.0, 0.0, 0.0],
						help="Desired normal direction vector (x y z) in world frame")
	parser.add_argument("--offset", type=float, default=0.0,
						help="Offset along normal to place paddle contact (meters)")
	parser.add_argument("--t-arrive", type=float, default=5.0,
						help="Seconds to reach the target pose")
	parser.add_argument("--hold-seconds", type=float, default=2.0,
						help="How long to hold the target pose after arrival")
	parser.add_argument("--no-viewer", action="store_true",
						help="Run headless without the MuJoCo viewer")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	# Inputs are interpreted in world coordinates.
	impact = np.array(args.impact, dtype=np.float64)
	desired = np.array(args.normal, dtype=np.float64)
	if float(np.linalg.norm(desired)) < 1e-9:
		raise ValueError("--normal must be a non-zero vector")
	desired = normalize_vector(desired)

	env = Environment(scene_xml="assets/scene.xml", randomize=False)
	ctrl_cfg = load_config("config/controller.yaml")
	pd_cfg = ctrl_cfg.get("pd_controller", {})
	kp = np.asarray(pd_cfg.get("kp", []), dtype=np.float64).reshape(-1)
	kd = np.asarray(pd_cfg.get("kd", []), dtype=np.float64).reshape(-1)
	if kp.shape[0] == env.n_dof and kd.shape[0] == env.n_dof:
		_apply_position_gains(env, kp, kd)
	ik = NumericalIKSolver(
		model=env.model,
		data=env.data,
		end_effector_body="paddle",
		end_effector_site="paddle_contact",
		position_weight=1.0,
		orientation_weight=0.25,
		max_iterations=500,
	)
	traj_gen = MinimumJerkTrajectory()

	env.reset(options={"randomize": False})
	env._park_ball()

	home_pos = env.home_position.copy()
	env.set_robot_joints(home_pos, np.zeros(env.n_dof))
	mujoco.mj_forward(env.model, env.data)

	t_arrive = float(args.t_arrive)
	target_pos = impact - desired * float(args.offset)
	q_goal, ik_ok = ik.solve(
		target_position=target_pos,
		target_normal=desired,
		initial_guess=home_pos,
	)
	
	ik_pos, ik_quat = ik.forward_kinematics(q_goal)
	ik_normal = _quat_to_z_axis(ik_quat)
	ik_pos_err = float(np.linalg.norm(ik_pos - target_pos))
	ik_ang_err = _angle_deg(ik_normal, desired)

	# Feasibility check: can the desired normal be achieved at the target position?
	pos_only_ik = NumericalIKSolver(
		model=env.model,
		data=env.data,
		end_effector_body="paddle",
		end_effector_site="paddle_contact",
		position_weight=1.0,
		orientation_weight=0.0,
		max_iterations=300,
	)
	q_pos, _ = pos_only_ik.solve(
		target_position=target_pos,
		initial_guess=home_pos,
	)
	pos_only_pos, pos_only_quat = pos_only_ik.forward_kinematics(q_pos)
	pos_only_normal = _quat_to_z_axis(pos_only_quat)
	pos_only_pos_err = float(np.linalg.norm(pos_only_pos - target_pos))
	pos_only_ang_err = _angle_deg(pos_only_normal, desired)

	traj = traj_gen.generate_trajectory(
		np.array([home_pos, q_goal]),
		np.array([0.0, t_arrive]),
		dt=env.dt,
	)
	traj_idx_arrive = min(int(round(t_arrive / env.dt)), len(traj) - 1)
	q_cmd_arrive = traj[traj_idx_arrive]["position"].copy()
	traj_goal_err = float(np.linalg.norm(q_cmd_arrive - q_goal))

	print("=" * 70)
	print("  PADDLE ALIGNMENT TEST")
	print("=" * 70)
	print(f"  impact        : {impact.tolist()}")
	print(f"  normal target : {desired.tolist()}")
	print(f"  offset        : {args.offset:.4f} m")
	print(f"  t_arrive      : {t_arrive:.3f} s")
	print(f"  IK            : {'converged' if ik_ok else 'approx'}")
	print(f"  IK target err : {ik_pos_err:.4f} m, normal {ik_ang_err:.2f} deg")
	print(f"  Traj goal err : {traj_goal_err:.6f} rad")
	if pos_only_ang_err > 20.0:
		print(f"  Note          : desired normal is ~{pos_only_ang_err:.1f} deg away at exact position")
		print(f"                  (pos-only IK error {pos_only_pos_err:.4f} m)")

	viewer = None
	interrupted = False
	try:
		if not args.no_viewer:
			viewer = mujoco.viewer.launch_passive(env.model, env.data)
			viewer.cam.lookat[:] = [0.6, 0.0, 1.0]
			viewer.cam.distance = 3.0
			viewer.cam.elevation = -20
			viewer.cam.azimuth = 55

		total_time = t_arrive + float(args.hold_seconds)
		total_steps = int(total_time / env.dt) + 1

		pos_err = None
		ang_err = None
		pos_err_vec = None
		normal_err_vec = None
		q_track_err = None

		for step in range(total_steps):
			if step < len(traj):
				ctrl = traj[step]["position"]
			else:
				ctrl = traj[-1]["position"]

			env._sim_step(ctrl)

			paddle_pos, _ = env.get_end_effector_pose()
			paddle_normal = env.get_paddle_normal()
			_update_debug_geoms(viewer, impact, desired, paddle_pos, paddle_normal)

			if viewer is not None and not _sync_viewer_safe(viewer, env.dt):
				viewer = None

			t = env.get_simulation_time()
			if pos_err is None and t >= t_arrive:
				paddle_pos, _ = env.get_end_effector_pose()
				paddle_normal = env.get_paddle_normal()
				pos_err = float(np.linalg.norm(paddle_pos - impact))
				ang_err = _angle_deg(paddle_normal, desired)
				pos_err_vec = paddle_pos - impact
				normal_err_vec = paddle_normal - desired
				q_actual = env.get_robot_state()["position"]
				q_track_err = float(np.linalg.norm(q_actual - q_cmd_arrive))
	except KeyboardInterrupt:
		interrupted = True
		print("\n  Interrupted: closing viewer and shutting down...")
	finally:
		if viewer is not None:
			try:
				viewer.close()
			except Exception:
				pass
		env.close()

	if interrupted:
		sys.stdout.flush()
		sys.stderr.flush()
		os._exit(0)

	if pos_err is None:
		paddle_pos, _ = env.get_end_effector_pose()
		paddle_normal = env.get_paddle_normal()
		pos_err = float(np.linalg.norm(paddle_pos - impact))
		ang_err = _angle_deg(paddle_normal, desired)
		pos_err_vec = paddle_pos - impact
		normal_err_vec = paddle_normal - desired

	print("\n  Alignment results")
	print(f"    position error : {pos_err:.4f} m")
	print(f"    normal error   : {ang_err:.2f} deg")
	if pos_err_vec is not None:
		print("    position error (x,y,z): "
			f"{pos_err_vec[0]:+.4f}, {pos_err_vec[1]:+.4f}, {pos_err_vec[2]:+.4f} m")
	if normal_err_vec is not None:
		print("    normal error (x,y,z)  : "
			f"{normal_err_vec[0]:+.4f}, {normal_err_vec[1]:+.4f}, {normal_err_vec[2]:+.4f}")
	if q_track_err is not None:
		print(f"    joint track err: {q_track_err:.4f} rad")


if __name__ == "__main__":
	main()
