"""
ControlPipeline
===============

Bridges the RL policy action → IK → trajectory → joint position commands
that can be passed to Environment.step().

Two modes
---------
task_space
    Policy action = 10-D vector:
        [impact_x, impact_y, impact_z,      (3) target Cartesian position
         t_impact,                           (1) seconds until impact
         nx, ny, nz,                         (3) desired paddle normal (unit vec)
         vx_impact, vy_impact, vz_impact]    (3) desired paddle velocity at impact
    Pipeline: IK(impact_pos, paddle_normal) → q_goal
              Trajectory(q_now → q_goal, T=t_impact)

joint_space
    Policy action = 7-D vector of normalised joint targets ∈ [-1, 1].
    Pipeline: de-normalise → Trajectory(q_now → q_target, T=t_exec)
              (no IK required)

Usage
-----
pipeline = ControlPipeline(
    env=env, ik_solver=ik, traj_gen=traj_gen,
    mode="task_space",
    home_pos=home_pos,
    action_repeat=20,          # sim steps per policy step (1 kHz / 20 = 50 Hz)
    t_exec_joint=0.4,          # fixed execution time for joint-space mode
)

# At the start of each policy step:
pipeline.plan(action, current_q)

# Inside the action_repeat inner loop:
ctrl = pipeline.get_ctrl(inner_step)   # (7,) joint position target
obs, rew, done, info = env.step(ctrl)
"""

import numpy as np
import typing as t
import mujoco
from src.planner.trajectory_planner import MinimumJerkTrajectory
from src.kinematics.inverse_kinematics import NumericalIKSolver, JacobianIKSolver
from src.planner.trajectory_planner import MinimumJerkTrajectory, CubicSplineTrajectory, BSplineTrajectory, TrapezoidalVelocityTrajectory

# Lazy imports so the module loads even when some deps are missing
_IK_TYPES = None
_TRAJ_TYPES = None

def _ensure_imports():
    global _IK_TYPES, _TRAJ_TYPES
    if _IK_TYPES is None:
        _IK_TYPES = NumericalIKSolver, JacobianIKSolver
    if _TRAJ_TYPES is None:
        _TRAJ_TYPES = MinimumJerkTrajectory, CubicSplineTrajectory, BSplineTrajectory, TrapezoidalVelocityTrajectory

# Robot workspace bounds (used to clip/validate IK targets)
WS_X = (0.60, 1.80)
WS_Y = (-0.70, 0.70)
WS_Z = (0.75, 1.80)

# Joint limits for the FR3 (radians)
Q_MIN = np.array([-2.9007, -1.8326, -2.9007, -3.0718, -2.8774, -0.0175, -3.0543])
Q_MAX = np.array([ 2.9007,  1.8326,  2.9007, -0.0698,  2.8774,  3.7525,  3.0543])

# Action space bounds for task-space normalisation (used in gym_env)
ACTION_LOW_TASK  = np.array([WS_X[0], WS_Y[0], WS_Z[0], 0.10, -1,-1,-1, -5,-5,-5], dtype=np.float32)
ACTION_HIGH_TASK = np.array([WS_X[1], WS_Y[1], WS_Z[1], 2.00,  1, 1, 1,  5, 5, 5], dtype=np.float32)


class ControlPipeline:
    """
    Integrates IK + trajectory planning into a single step-by-step interface compatible with Environment.step().

    The pipeline is replanned every time plan() is called (i.e. every policy step = every action_repeat sim steps).
    """

    def __init__(self, env, ik_solver=None, traj_gen=None, mode: str = "task_space", home_pos: t.Optional[np.ndarray] = None, action_repeat: int = 20, t_exec_joint: float = 0.40, dt: t.Optional[float] = None):
        """
        Parameters
        ----------
        env : Environment
            The simulation environment instance.
        ik_solver : NumericalIKSolver | JacobianIKSolver | None
            Required for task_space mode; ignored in joint_space mode.
        traj_gen : BaseTrajectoryGenerator | None
            Trajectory generator; if None, MinimumJerkTrajectory is used.
        mode : str
            "task_space" or "joint_space".
        home_pos : np.ndarray (7,)
            Robot home joint configuration used as IK initial guess.
        action_repeat : int
            Number of sim steps per policy step.
        t_exec_joint : float
            Execution time (seconds) in joint-space mode.
        dt : float | None
            Simulation timestep; defaults to env.dt.
        """
        _ensure_imports()

        self.env = env
        self.ik_solver = ik_solver
        self.mode = mode
        self.action_repeat = action_repeat
        self.t_exec_joint = t_exec_joint
        self.dt = dt if dt is not None else env.dt
        self.home_pos = home_pos.copy() if home_pos is not None else env.home_position.copy()

        # Trajectory generator — default to MinimumJerk
        if traj_gen is None:
            self.traj_gen = MinimumJerkTrajectory()
        else:
            self.traj_gen = traj_gen

        # Cache whether the generator supports the max_steps kwarg so we don't
        # call inspect.signature() inside the hot planning loop every step.
        import inspect as _inspect
        try:
            _sig = _inspect.signature(self.traj_gen.generate_trajectory)
            self._traj_supports_max_steps = "max_steps" in _sig.parameters
        except Exception:
            self._traj_supports_max_steps = False

        # Internal trajectory buffer
        self._traj: t.List[t.Dict] = []
        self._plan_ok = False
        self._q_goal = self.home_pos.copy()
        self._t_impact = 0.0

        # Decoded action components (informational, exposed for reward calc)
        self.last_impact_pos: t.Optional[np.ndarray] = None
        self.last_paddle_normal: t.Optional[np.ndarray] = None
        self.last_impact_vel: t.Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, action: np.ndarray, current_q: np.ndarray) -> bool:
        """
        Decode the policy action and pre-compute the trajectory.

        Parameters
        ----------
        action : np.ndarray
            task_space: (10,) raw action
            joint_space: (7,) normalised joint targets ∈ [-1, 1]
        current_q : np.ndarray (7,)
            Current joint positions.

        Returns
        -------
        bool
            True if IK converged (task_space) or always True (joint_space).
        """
        if self.mode == "task_space":
            return self._plan_task(action, current_q)
        else:
            return self._plan_joint(action, current_q)

    def get_ctrl(self, inner_step: int) -> np.ndarray:
        """
        Return the joint position command for the given inner loop step.

        Parameters
        ----------
        inner_step : int
            Index within the action_repeat window [0, action_repeat).

        Returns
        -------
        np.ndarray (7,)
            Joint position targets for env.step().
        """
        if not self._plan_ok or len(self._traj) == 0:
            return self.home_pos.copy()

        idx = min(inner_step, len(self._traj) - 1)
        return self._traj[idx]["position"].copy()

    def reset(self):
        """Clear trajectory buffer (call at episode start)."""
        self._traj = []
        self._plan_ok = False
        self._q_goal = self.home_pos.copy()
        self.last_impact_pos = None
        self.last_paddle_normal = None
        self.last_impact_vel = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _plan_task(self, action: np.ndarray, current_q: np.ndarray) -> bool:
        """IK + trajectory for task-space action."""
        # --- Decode action ---
        impact_pos    = np.clip(action[0:3],
                                [WS_X[0], WS_Y[0], WS_Z[0]],
                                [WS_X[1], WS_Y[1], WS_Z[1]])
        t_impact      = float(np.clip(action[3], 0.05, 3.0))
        impact_vel = action[7:10].copy()
        impact_speed = float(np.linalg.norm(impact_vel))
        if impact_speed > 1e-6:
            impact_dir = impact_vel / impact_speed
            # Match comprehensive test behavior: paddle normal should oppose
            # incoming ball direction at impact.
            paddle_normal = -impact_dir
        else:
            # Fallback to explicit normal from action when impact velocity is
            # unavailable or degenerate.
            paddle_normal = action[4:7].copy()
            n_norm = np.linalg.norm(paddle_normal)
            if n_norm > 1e-6:
                paddle_normal /= n_norm
            else:
                paddle_normal = np.array([1.0, 0.0, 0.0])

        self.last_impact_pos    = impact_pos
        self.last_paddle_normal = paddle_normal
        self.last_impact_vel    = impact_vel
        self._t_impact          = t_impact

        # Target the paddle contact site slightly behind the ball center at
        # impact, so the blade face reaches the ball instead of proximal links.
        ball_r = float(getattr(self.env, "ball_radius", 0.02))
        contact_offset = ball_r + self._PADDLE_BLADE_HALF_THICKNESS + self._CONTACT_MARGIN
        ee_target_pos = impact_pos - paddle_normal * contact_offset

        # --- IK ---
        if self.ik_solver is not None:
            q_goal, ik_ok = self.ik_solver.solve(
                target_position=ee_target_pos,
                target_normal=paddle_normal,
                initial_guess=current_q,
            )
        else:
            # No IK: hold home position
            q_goal = self.home_pos.copy()
            ik_ok  = False

        # Clamp per-joint delta so the trajectory T stays short enough for the
        # robot to execute meaningful motion within the action_repeat window.
        # Receding-horizon replanning (every policy step) lets the arm converge
        # toward the IK target over multiple steps.
        q_goal = np.clip(q_goal, current_q - self._DELTA_Q_MAX, current_q + self._DELTA_Q_MAX)
        q_goal = np.clip(q_goal, Q_MIN + 0.01, Q_MAX - 0.01)
        # Note: the static torque-ratio fallback (mj_forward + mj_inverse per
        # step) was removed for RL training throughput.  The instability guard
        # in env.step() (qacc > 1e5) already terminates episodes that reach
        # physically infeasible poses, so the fallback is not needed here.

        self._q_goal = q_goal

        # --- Trajectory ---
        T = max(t_impact, self.dt * self.action_repeat)
        self._generate_traj(current_q, q_goal, T)
        self._plan_ok = True
        return ik_ok


    def _plan_joint(self, action: np.ndarray, current_q: np.ndarray) -> bool:
        """Direct joint-space delta action (normalised ∈ [-1,1] → joint delta).

        Skips trajectory planning entirely.  q_goal is held as the position
        target for every inner sim step so the position controller (kp=200,
        kd≈31 Nm·s/rad) has the full action_repeat window (20 ms) to build
        velocity toward the target, rather than chasing a near-zero ramp for
        the first 15 of 20 inner steps (MinimumJerk flat start).

        With _DELTA_Q_MAX = 0.25 rad/step the controller reaches roughly
        0.02–0.06 rad actual movement per step depending on joint inertia,
        giving clear visible arm motion at 50 Hz while staying within FR3
        velocity limits in simulation.
        """
        delta  = action * self._DELTA_Q_MAX
        q_goal = np.clip(current_q + delta, Q_MIN + 0.01, Q_MAX - 0.01)
        self._q_goal = q_goal

        # Broadcast q_goal across all inner steps — no trajectory overhead.
        _zero = np.zeros_like(q_goal)
        traj_pt = {"position": q_goal, "velocity": _zero, "acceleration": _zero}
        self._traj = [traj_pt] * self.action_repeat
        self._plan_ok = True
        return True

    # Maximum safe joint speed (rad/s) — used to enforce minimum trajectory time
    # FR3 joint speed limits: ~2.62 rad/s, use 50% of that to stay stable
    _MAX_SAFE_JOINT_SPEED = 1.0
    # Maximum joint delta per policy step (joint_space mode: delta actions;
    # task_space mode: per-joint IK output clamping).
    # 0.25 rad/step: controller (kp=200, kd≈31) reaches ~0.02–0.06 rad actual
    # displacement per step depending on joint inertia — visible arm motion at
    # 50 Hz.  Implied velocity (0.25×50=12.5 rad/s) is a commanded reference;
    # actual joint velocity is bounded by the controller response and inertia,
    # staying within FR3 limits in simulation.
    _DELTA_Q_MAX = 0.25
    _PADDLE_BLADE_HALF_THICKNESS = 0.00325
    _CONTACT_MARGIN = 0.002
    _MAX_STATIC_TORQUE_RATIO = 0.95

    def _generate_traj(self, q_start: np.ndarray, q_goal: np.ndarray, T: float):
        """Generate a joint-space trajectory at simulation dt resolution.

        The full trajectory from q_start → q_goal spans T seconds.  Only the
        first ``action_repeat`` points (one per inner sim step = dt seconds
        each) are kept in ``self._traj`` so that ``get_ctrl(inner_step)``
        delivers a correctly-timed position target every 1 ms, rather than
        jumping through T/(action_repeat-1) seconds of motion in a single step.
        """
        # Enforce a minimum T so that commanded joint velocities stay physical.
        max_delta = float(np.max(np.abs(q_goal - q_start)))
        T_min_kinematic = max_delta / self._MAX_SAFE_JOINT_SPEED
        T_min_action    = self.dt * self.action_repeat
        T = max(T, T_min_kinematic, T_min_action)

        waypoints = np.stack([q_start, q_goal])
        times     = np.array([0.0, T])
        # Generate at 1 ms (dt) resolution so each sim step receives the
        # trajectory point that is correct for that moment in time.
        # Pass max_steps so generators that support it (e.g. MinimumJerkTrajectory)
        # compute only the action_repeat points we will actually use, avoiding
        # the O(T/dt) work needed to generate then discard the rest of the path.
        gen_kwargs: dict = {"max_steps": self.action_repeat} if self._traj_supports_max_steps else {}

        try:
            full = self.traj_gen.generate_trajectory(waypoints, times, self.dt, **gen_kwargs)
        except Exception:
            # Fallback: linear interpolation at dt resolution
            n = self.action_repeat
            full = []
            for i in range(n):
                alpha = i / max(n - 1, 1)
                full.append({"position": q_start + alpha * (q_goal - q_start),
                             "velocity": np.zeros(len(q_start)),
                             "acceleration": np.zeros(len(q_start))})
        # Retain only the first action_repeat steps: the portion of the
        # trajectory that this policy step will execute.
        self._traj = full[:self.action_repeat]

    def _estimate_static_torque_ratio(self, q_goal: np.ndarray) -> float:
        """Estimate hold-torque demand ratio (required / actuator limit) at q_goal."""
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()
        saved_qacc = self.env.data.qacc.copy()
        try:
            self.env.data.qpos[:self.env.n_dof] = q_goal
            self.env.data.qvel[:self.env.n_dof] = 0.0
            self.env.data.qacc[:self.env.n_dof] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            mujoco.mj_inverse(self.env.model, self.env.data)

            tau_req = self.env.data.qfrc_inverse[:self.env.n_dof]
            ratio_max = 0.0
            for i in range(self.env.n_dof):
                if int(self.env.model.actuator_forcelimited[i]) == 0:
                    continue
                f_max = float(max(
                    abs(self.env.model.actuator_forcerange[i, 0]),
                    abs(self.env.model.actuator_forcerange[i, 1]),
                    1e-6,
                ))
                ratio = abs(float(tau_req[i])) / f_max
                if ratio > ratio_max:
                    ratio_max = ratio
            return ratio_max
        finally:
            self.env.data.qpos[:] = saved_qpos
            self.env.data.qvel[:] = saved_qvel
            self.env.data.qacc[:] = saved_qacc
            mujoco.mj_forward(self.env.model, self.env.data)
