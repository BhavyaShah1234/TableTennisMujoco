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


def _quat_to_face_normal(quat: np.ndarray) -> np.ndarray:
	"""Return the paddle face normal from a site quaternion.
	Face normal = body -Y axis (after Rx(90°) applied to visual mesh so
	handle connects to link7; STL +Z face normal maps to body -Y)."""
	mat = np.zeros(9, dtype=np.float64)
	mujoco.mju_quat2Mat(mat, quat)
	return -mat.reshape(3, 3)[:, 1]  # negative Y column


def _alignment_snapshot(env: Environment, impact: np.ndarray, desired: np.ndarray) -> dict:
	paddle_pos, _ = env.get_end_effector_pose()
	paddle_normal = env.get_paddle_normal()
	return {
		"paddle_pos": paddle_pos,
		"paddle_normal": paddle_normal,
		"pos_err": float(np.linalg.norm(paddle_pos - impact)),
		"ang_err": _angle_deg(paddle_normal, desired),
		"pos_err_vec": paddle_pos - impact,
		"normal_err_vec": paddle_normal - desired,
	}


def _min_joint_limit_margin(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> float:
	if q.shape[0] == 0:
		return 0.0
	return float(np.min(np.minimum(q - q_min, q_max - q)))


def _estimate_static_torque_ratio(env: Environment, q_goal: np.ndarray) -> float:
	# Estimate gravity/constraint hold torque at q_goal via inverse dynamics.
	# Compares required torque against the TOTAL force capacity for each DOF,
	# summing position actuator + motor actuator force limits (both are used:
	# position actuators handle errors; motor actuators handle gravity via
	# qfrc_bias feed-forward added in _sim_step).
	saved_qpos = env.data.qpos.copy()
	saved_qvel = env.data.qvel.copy()
	saved_qacc = env.data.qacc.copy()
	try:
		env.data.qpos[:env.n_dof] = q_goal
		env.data.qvel[:env.n_dof] = 0.0
		env.data.qacc[:env.n_dof] = 0.0
		mujoco.mj_forward(env.model, env.data)
		mujoco.mj_inverse(env.model, env.data)
		tau_req = env.data.qfrc_inverse[:env.n_dof].copy()
		rat_max = 0.0
		for i in range(env.n_dof):
			# Sum force limits from position actuator (i) AND motor actuator (n_dof+i).
			# Position actuators use forcelimited/forcerange; motor actuators use
			# ctrllimited/ctrlrange (gear=1 so ctrlrange == force range).
			f_total = 0.0
			for act_i in (i, env.n_dof + i):
				if act_i >= env.model.nu:
					break
				if int(env.model.actuator_forcelimited[act_i]):
					f_total += float(max(abs(env.model.actuator_forcerange[act_i, 0]),
					                     abs(env.model.actuator_forcerange[act_i, 1]), 1e-6))
				elif int(env.model.actuator_ctrllimited[act_i]):
					# For gear=1 motor: max force = max(|ctrlrange|)
					gear = float(abs(env.model.actuator_gear[act_i, 0])) or 1.0
					f_total += gear * float(max(abs(env.model.actuator_ctrlrange[act_i, 0]),
					                            abs(env.model.actuator_ctrlrange[act_i, 1]), 1e-6))
			if f_total < 1e-9:
				continue
			rat = abs(float(tau_req[i])) / f_total
			if rat > rat_max:
				rat_max = rat
		return rat_max
	finally:
		env.data.qpos[:] = saved_qpos
		env.data.qvel[:] = saved_qvel
		env.data.qacc[:] = saved_qacc
		mujoco.mj_forward(env.model, env.data)


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
	parser.add_argument("--normal", nargs=3, type=float, default=[-1.0, 1.0, -1.0],
						help="Desired normal direction vector (x y z) in world frame")
	parser.add_argument("--offset", type=float, default=0.0,
						help="Offset along normal to place paddle contact (meters)")
	parser.add_argument("--ik-position-weight", type=float, default=1.0,
						help="IK weight for position term")
	parser.add_argument("--ik-orientation-weight", type=float, default=0.8,
						help="IK weight for normal-alignment term")
	parser.add_argument("--t-arrive", type=float, default=5.0,
						help="Seconds to reach the target pose")
	parser.add_argument("--hold-seconds", type=float, default=2.0,
						help="How long to hold the target pose after arrival")
	parser.add_argument("--apply-config-gains", action="store_true",
						help="Override scene position-actuator gains with config/controller.yaml")
	parser.add_argument("--refine-iters", type=int, default=0,
						help="Extra closed-loop IK correction steps after hold")
	parser.add_argument("--refine-alpha", type=float, default=0.35,
						help="Blend factor for each refinement step (0,1]")
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
	if args.apply_config_gains:
		ctrl_cfg = load_config("config/controller.yaml")
		pd_cfg = ctrl_cfg.get("pd_controller", {})
		kp = np.asarray(pd_cfg.get("kp", []), dtype=np.float64).reshape(-1)
		kd = np.asarray(pd_cfg.get("kd", []), dtype=np.float64).reshape(-1)
		if kp.shape[0] == env.n_dof and kd.shape[0] == env.n_dof:
			_apply_position_gains(env, kp, kd)
			print("  Using config/controller.yaml gains for position actuators")
		else:
			print("  Config gains shape mismatch; using scene actuator gains")
	ik = NumericalIKSolver(
		model=env.model,
		data=env.data,
		end_effector_body="paddle",
		end_effector_site="paddle_contact",
		end_effector_normal_site="paddle_normal",
		position_weight=float(args.ik_position_weight),
		orientation_weight=float(args.ik_orientation_weight),
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
	q_goal_limit_margin = _min_joint_limit_margin(q_goal, ik.q_min, ik.q_max)
	static_torque_ratio = _estimate_static_torque_ratio(env, q_goal)
	
	ik_pos, ik_quat = ik.forward_kinematics(q_goal)
	ik_normal = _quat_to_face_normal(ik_quat)
	ik_pos_err = float(np.linalg.norm(ik_pos - target_pos))
	ik_ang_err = _angle_deg(ik_normal, desired)

	# Feasibility check: can the desired normal be achieved at the target position?
	pos_only_ik = NumericalIKSolver(
		model=env.model,
		data=env.data,
		end_effector_body="paddle",
		end_effector_site="paddle_contact",
		end_effector_normal_site="paddle_normal",
		position_weight=1.0,
		orientation_weight=0.0,
		max_iterations=300,
	)
	q_pos, _ = pos_only_ik.solve(
		target_position=target_pos,
		initial_guess=home_pos,
	)
	pos_only_pos, pos_only_quat = pos_only_ik.forward_kinematics(q_pos)
	pos_only_normal = _quat_to_face_normal(pos_only_quat)
	pos_only_pos_err = float(np.linalg.norm(pos_only_pos - target_pos))
	pos_only_ang_err = _angle_deg(pos_only_normal, desired)

	# Auto-scale t_arrive so no joint exceeds V_LIM_RAD_S at peak trajectory
	# velocity (min-jerk peak = |Δq| * 1.875 / T).  Prevents the 500 Hz
	# Nyquist oscillation from growing large enough to destabilise joint 7.
	_V_LIM = 2.0  # rad/s conservative limit (FR3 max is 2.61 rad/s)
	_delta_q = np.abs(q_goal - home_pos)
	_t_vel_limit = float(np.max(_delta_q * 1.875 / _V_LIM)) if np.any(_delta_q > 1e-6) else 0.0
	if _t_vel_limit > t_arrive:
		print(f"  Auto-scaling t_arrive: {t_arrive:.2f}s → {_t_vel_limit:.2f}s "
			  f"(joint velocity limit {_V_LIM} rad/s)")
		t_arrive = _t_vel_limit

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
	print(f"  IK weights    : pos={float(args.ik_position_weight):.3f}, ori={float(args.ik_orientation_weight):.3f}")
	print(f"  IK            : {'converged' if ik_ok else 'approx'}")
	print(f"  IK target err : {ik_pos_err:.4f} m, normal {ik_ang_err:.2f} deg")
	print(f"  Traj goal err : {traj_goal_err:.6f} rad")
	print(f"  q_goal margin : {q_goal_limit_margin:.4f} rad to nearest joint limit")
	print(f"  q_goal static torque ratio : {static_torque_ratio:.3f} (<=1 is hold-feasible)")
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

		arrival_metrics = None
		best_pos_err = np.inf
		best_ang_err = np.inf
		best_t = 0.0
		q_track_err_arrive = None
		max_act_sat = 0.0
		min_q_margin_runtime = np.inf

		for step in range(total_steps):
			if step < len(traj):
				ctrl = traj[step]["position"]
			else:
				ctrl = traj[-1]["position"]

			env._sim_step(ctrl)

			paddle_pos, _ = env.get_end_effector_pose()
			paddle_normal = env.get_paddle_normal()
			_update_debug_geoms(viewer, impact, desired, paddle_pos, paddle_normal)

			# Runtime stress metrics: actuator saturation and joint-limit proximity.
			for i in range(env.n_dof):
				if int(env.model.actuator_forcelimited[i]) == 0:
					continue
				f_max = float(max(abs(env.model.actuator_forcerange[i, 0]), abs(env.model.actuator_forcerange[i, 1]), 1e-6))
				rat = abs(float(env.data.actuator_force[i])) / f_max
				if rat > max_act_sat:
					max_act_sat = rat

			q_actual_runtime = env.get_robot_state()["position"]
			q_margin = _min_joint_limit_margin(q_actual_runtime, ik.q_min, ik.q_max)
			if q_margin < min_q_margin_runtime:
				min_q_margin_runtime = q_margin

			snap = _alignment_snapshot(env, impact, desired)
			t = env.get_simulation_time()
			if snap["pos_err"] < best_pos_err:
				best_pos_err = snap["pos_err"]
				best_ang_err = snap["ang_err"]
				best_t = t

			if viewer is not None and not _sync_viewer_safe(viewer, env.dt):
				viewer = None

			if arrival_metrics is None and t >= t_arrive:
				arrival_metrics = snap
				q_actual = env.get_robot_state()["position"]
				q_track_err_arrive = float(np.linalg.norm(q_actual - q_cmd_arrive))

		# Optional final local refinement: repeatedly re-solve IK from current q.
		if args.refine_iters > 0:
			alpha = float(np.clip(args.refine_alpha, 1e-3, 1.0))
			for _ in range(int(args.refine_iters)):
				q_now = env.get_robot_state()["position"]
				q_ref, _ = ik.solve(
					target_position=target_pos,
					target_normal=desired,
					initial_guess=q_now,
				)
				ctrl = q_now + alpha * (q_ref - q_now)
				env._sim_step(ctrl)

				paddle_pos, _ = env.get_end_effector_pose()
				paddle_normal = env.get_paddle_normal()
				_update_debug_geoms(viewer, impact, desired, paddle_pos, paddle_normal)
				if viewer is not None and not _sync_viewer_safe(viewer, env.dt):
					viewer = None

				snap = _alignment_snapshot(env, impact, desired)
				t = env.get_simulation_time()
				if snap["pos_err"] < best_pos_err:
					best_pos_err = snap["pos_err"]
					best_ang_err = snap["ang_err"]
					best_t = t

				for i in range(env.n_dof):
					if int(env.model.actuator_forcelimited[i]) == 0:
						continue
					f_max = float(max(abs(env.model.actuator_forcerange[i, 0]), abs(env.model.actuator_forcerange[i, 1]), 1e-6))
					rat = abs(float(env.data.actuator_force[i])) / f_max
					if rat > max_act_sat:
						max_act_sat = rat
				q_actual_runtime = env.get_robot_state()["position"]
				q_margin = _min_joint_limit_margin(q_actual_runtime, ik.q_min, ik.q_max)
				if q_margin < min_q_margin_runtime:
					min_q_margin_runtime = q_margin
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

	final_metrics = _alignment_snapshot(env, impact, desired)
	q_final = env.get_robot_state()["position"]
	q_track_err_final = float(np.linalg.norm(q_final - q_goal))
	q_final_limit_margin = _min_joint_limit_margin(q_final, ik.q_min, ik.q_max)

	if arrival_metrics is None:
		arrival_metrics = final_metrics
		q_track_err_arrive = float(np.linalg.norm(q_final - q_cmd_arrive))
	if q_track_err_arrive is None:
		q_track_err_arrive = float(np.linalg.norm(q_final - q_cmd_arrive))

	if not np.isfinite(min_q_margin_runtime):
		min_q_margin_runtime = q_final_limit_margin

	print("\n  Alignment results")
	print("    at arrival")
	print(f"      position error : {arrival_metrics['pos_err']:.4f} m")
	print(f"      normal error   : {arrival_metrics['ang_err']:.2f} deg")
	if arrival_metrics.get("pos_err_vec") is not None:
		print("    position error (x,y,z): "
			f"{arrival_metrics['pos_err_vec'][0]:+.4f}, {arrival_metrics['pos_err_vec'][1]:+.4f}, {arrival_metrics['pos_err_vec'][2]:+.4f} m")
	if arrival_metrics.get("normal_err_vec") is not None:
		print("    normal error (x,y,z)  : "
			f"{arrival_metrics['normal_err_vec'][0]:+.4f}, {arrival_metrics['normal_err_vec'][1]:+.4f}, {arrival_metrics['normal_err_vec'][2]:+.4f}")
	print(f"      joint track err to q_cmd(t_arrive): {q_track_err_arrive:.4f} rad")

	print("\n    at end (after hold/refine)")
	print(f"      position error : {final_metrics['pos_err']:.4f} m")
	print(f"      normal error   : {final_metrics['ang_err']:.2f} deg")
	print("      position error (x,y,z): "
		f"{final_metrics['pos_err_vec'][0]:+.4f}, {final_metrics['pos_err_vec'][1]:+.4f}, {final_metrics['pos_err_vec'][2]:+.4f} m")
	print("      normal error (x,y,z)  : "
		f"{final_metrics['normal_err_vec'][0]:+.4f}, {final_metrics['normal_err_vec'][1]:+.4f}, {final_metrics['normal_err_vec'][2]:+.4f}")
	print(f"      joint track err to q_goal: {q_track_err_final:.4f} rad")

	print("\n    over full window")
	print(f"      best position error: {best_pos_err:.4f} m at t={best_t:.3f} s")
	print(f"      normal error at best-pos time: {best_ang_err:.2f} deg")
	print(f"      max actuator saturation ratio: {max_act_sat:.3f} (1.0 means force limit)")
	print(f"      minimum runtime joint-limit margin: {min_q_margin_runtime:.4f} rad")
	print(f"      final joint-limit margin: {q_final_limit_margin:.4f} rad")


if __name__ == "__main__":
	main()
