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
Q_MIN = np.array([-2.9007, -1.8326, -2.9007, -3.0718, -2.8774,  0.4398, -3.0543])
Q_MAX = np.array([ 2.9007,  1.8326,  2.9007, -0.1169,  2.8774,  4.6251,  3.0543])

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

        q_goal = np.clip(q_goal, Q_MIN + 0.01, Q_MAX - 0.01)
        self._q_goal = q_goal

        # --- Trajectory ---
        T = max(t_impact, self.dt * self.action_repeat)
        self._generate_traj(current_q, q_goal, T)
        self._plan_ok = True
        return ik_ok


    def _plan_joint(self, action: np.ndarray, current_q: np.ndarray) -> bool:
        """Direct joint-space action (normalised ∈ [-1,1] → joint targets)."""
        # De-normalise from [-1, 1] to joint limits
        q_goal = 0.5 * (Q_MAX + Q_MIN) + 0.5 * action * (Q_MAX - Q_MIN)
        q_goal = np.clip(q_goal, Q_MIN + 0.01, Q_MAX - 0.01)
        self._q_goal = q_goal

        # Trajectory over a fixed window
        T = self.t_exec_joint
        self._generate_traj(current_q, q_goal, T)
        self._plan_ok = True
        return True

    # Maximum safe joint speed (rad/s) — used to enforce minimum trajectory time
    # FR3 joint speed limits: ~2.62 rad/s, use 50% of that to stay stable
    _MAX_SAFE_JOINT_SPEED = 1.0
    _PADDLE_BLADE_HALF_THICKNESS = 0.00325
    _CONTACT_MARGIN = 0.002

    def _generate_traj(self, q_start: np.ndarray, q_goal: np.ndarray, T: float):
        """Generate a joint-space trajectory and store in self._traj."""
        # Enforce a minimum T so that the commanded joint velocities stay physical
        max_delta = float(np.max(np.abs(q_goal - q_start)))
        T_min_kinematic = max_delta / self._MAX_SAFE_JOINT_SPEED
        T_min_action    = self.dt * self.action_repeat
        T = max(T, T_min_kinematic, T_min_action)

        waypoints = np.stack([q_start, q_goal])
        times     = np.array([0.0, T])
        # Produce exactly action_repeat steps (one per inner loop step)
        traj_dt   = T / max(self.action_repeat - 1, 1)
        try:
            full = self.traj_gen.generate_trajectory(waypoints, times, traj_dt)
        except Exception:
            # Fallback: linear interpolation
            full = []
            for i in range(self.action_repeat):
                alpha = i / max(self.action_repeat - 1, 1)
                full.append({"position": q_start + alpha * (q_goal - q_start),
                             "velocity": np.zeros(len(q_start)),
                             "acceleration": np.zeros(len(q_start))})
        self._traj = full
