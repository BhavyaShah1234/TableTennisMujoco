"""
TableTennisEnv — self-contained Gymnasium environment for the MuJoCo table-tennis robot.
=========================================================================================

All MuJoCo simulation logic (previously in src.simulation.environment.Environment) is
merged directly into this class, eliminating the redundant wrapper layer.

Gymnasium registration
----------------------
The environment is registered under the id ``"TableTennis-v0"`` at import time via
``gymnasium.register()``.  Use either:

    import gymnasium
    import src.rl.gym_env          # triggers registration
    env = gymnasium.make("TableTennis-v0")

or the factory helper (works without prior ``gymnasium.make``):

    from src.rl.gym_env import make_env
    env = make_env(mode="task_space", ik_name="NumericalIKSolver",
                   traj_name="MinimumJerkTrajectory")

Raw simulation stepping (test / non-RL use)
-------------------------------------------
Call ``env._sim_step(ctrl)`` with a ``(7,)`` joint-position target to advance
exactly one MuJoCo timestep without going through the RL action pipeline.
Returns ``(obs_23d, 0.0, done, info)`` — the same 4-tuple as the legacy
``src.simulation.environment.Environment.step()``.
"""

from __future__ import annotations

import time
import typing as t
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

from src.rl.control_pipeline import ControlPipeline, Q_MIN, Q_MAX, ACTION_LOW_TASK, ACTION_HIGH_TASK
from src.rl.reward import RewardCalculator
from src.utils.utils import load_config


# ── Observation constants ────────────────────────────────────────────────────
OBS_DIM       = 19   # [ball_pos(3) ball_vel(3) ball_spin(3) joint_pos(7) paddle_normal(3)]
ACT_DIM_TASK  = 10
ACT_DIM_JOINT =  7

OBS_LOW  = np.full(OBS_DIM, -10.0, dtype=np.float32)
OBS_HIGH = np.full(OBS_DIM,  10.0, dtype=np.float32)
OBS_LOW[0:3]   = np.array([-2.0, -1.5, -0.5], dtype=np.float32)
OBS_HIGH[0:3]  = np.array([ 2.0,  1.5,  3.0], dtype=np.float32)
OBS_LOW[16:19] = OBS_HIGH[16:19] = None  # reset before assigning
OBS_LOW[16:19]  = -1.0   # paddle normal is a unit vector ∈ [-1, 1]
OBS_HIGH[16:19] =  1.0


# ── IK / trajectory builder helpers (used by __init__ and make_env) ──────────

def _build_ik(name: t.Optional[str], model, data) -> t.Optional[object]:
    """Instantiate an IK solver by name, using the given MuJoCo model/data."""
    if name is None or name == "None":
        return None
    from src.kinematics.inverse_kinematics import NumericalIKSolver, JacobianIKSolver
    common = dict(model=model, data=data, end_effector_body="paddle", end_effector_site="paddle_contact")
    if name == "NumericalIKSolver":
        return NumericalIKSolver(**common, position_weight=1.0,
                                 orientation_weight=0.25, max_iterations=500)
    if name == "JacobianIKSolver":
        return JacobianIKSolver(**common, step_size=0.1,
                                max_iterations=500, tolerance=5e-3)
    raise ValueError(f"Unknown IK solver: {name!r}")


def _build_traj(name: t.Optional[str]) -> t.Optional[object]:
    """Instantiate a trajectory generator by name."""
    if name is None or name == "None":
        return None
    from src.planner.trajectory_planner import (
        MinimumJerkTrajectory, CubicSplineTrajectory,
        BSplineTrajectory, TrapezoidalVelocityTrajectory,
    )
    if name == "TrapezoidalVelocityTrajectory":
        return TrapezoidalVelocityTrajectory(max_velocity=2.0, max_acceleration=5.0)
    MAP = {
        "MinimumJerkTrajectory": MinimumJerkTrajectory,
        "CubicSplineTrajectory": CubicSplineTrajectory,
        "BSplineTrajectory":     BSplineTrajectory,
    }
    if name not in MAP:
        raise ValueError(f"Unknown trajectory planner: {name!r}")
    return MAP[name]()


# ─────────────────────────────────────────────────────────────────────────────

class Environment(gym.Env):
    """
    Self-contained Gymnasium environment for the FR3 table-tennis robot.

    Merges MuJoCo simulation management and the RL interface into a single
    class.  Follows the standard ``gymnasium.Env`` API and can be registered
    with ``gymnasium.make("TableTennis-v0")``.

    Parameters
    ----------
    mode : str
        ``"task_space"`` (10-D action) or ``"joint_space"`` (7-D action).
    ik_solver : optional
        Pre-built IK solver instance.  Mutually exclusive with ``ik_name``.
    ik_name : str | None
        Name of the IK solver to build automatically.  Used when ``ik_solver``
        is ``None`` and ``mode == "task_space"``.
    traj_gen : optional
        Pre-built trajectory generator instance.  Mutually exclusive with
        ``traj_name``.
    traj_name : str | None
        Name of the trajectory generator to build automatically.
    action_repeat : int
        MuJoCo timesteps per policy step (default 20 → 50 Hz at 1 kHz sim).
    randomize : bool
        Randomise ball spawn on ``reset()``.
    render_mode : str | None
        ``"human"`` launches a MuJoCo passive viewer; ``None`` is headless.
    scene_xml : str
        Path to the MuJoCo scene XML.
    sim_cfg_path : str
        Path to ``config/simulation.yaml``.
    robot_cfg_path : str
        Path to ``config/robot.yaml``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, mode: str = "task_space", ik_solver: None = None, ik_name: t.Optional[str] = None, traj_gen: None = None, traj_name: t.Optional[str] = None, action_repeat: int = 20, randomize: bool = True, render_mode: t.Optional[str] = None, scene_xml: str = "assets/scene.xml", sim_cfg_path: str = "config/simulation.yaml", robot_cfg_path: str = "config/robot.yaml"):
        super(Environment, self).__init__()

        # ── Load configuration ────────────────────────────────────────────
        sim_cfg   = load_config(sim_cfg_path)
        robot_cfg = load_config(robot_cfg_path)

        # ── MuJoCo model & data ───────────────────────────────────────────
        self.model  = mujoco.MjModel.from_xml_path(scene_xml)
        base_pos = np.asarray(robot_cfg.get("robot", {}).get("base_position", []), dtype=np.float64).reshape(-1)
        if base_pos.size > 0:
            if base_pos.shape[0] != 3:
                raise ValueError("config/robot.yaml must define robot.base_position with 3 values")
            base_body_id = self.model.body("fr3_link0").id
            self.model.body_pos[base_body_id] = base_pos
        self.data   = mujoco.MjData(self.model)
        self.config = sim_cfg
        self.dt     = float(self.model.opt.timestep)
        self.n_dof  = 7  # FR3 has 7 joints
        self._paddle_contact_site_id = self.model.site("paddle_contact").id
        self._paddle_normal_site_id = self.model.site("paddle_normal").id

        # ── Ball ──────────────────────────────────────────────────────────
        self.ball_config  = sim_cfg["ball"]
        self._tested_ball_states = self._load_tested_ball_states(self.ball_config)
        self.ball_radius = float(self.ball_config.get("radius", 0.02))
        self._floor_geom_names = {"floor"}
        self.ball_body_id: t.Optional[int] = None
        self._setup_ball()

        # ── Termination thresholds ────────────────────────────────────────
        self.max_episode_time       = float(self.ball_config["max_episode_time"])
        self.min_velocity_threshold = float(self.ball_config["min_velocity_threshold"])
        self.ground_z_threshold     = float(self.ball_config["ground_z_threshold"])
        self.out_of_bounds_dist     = float(self.ball_config["out_of_bounds_distance"])

        # ── Raw-sim episode counters (used by _sim_step / _check_termination)
        self.episode_time  = 0.0
        self.episode_steps = 0

        # ── Robot home/ready configuration ────────────────────────────────
        robot_section = robot_cfg.get("robot", {})

        cfg_home = np.asarray(robot_section.get("home_position", []), dtype=np.float64).reshape(-1)
        if cfg_home.shape[0] != self.n_dof:
            raise ValueError(
                f"config/robot.yaml must define robot.home_position with {self.n_dof} values"
            )
        self.home_position = cfg_home.copy()

        # ── RL configuration ──────────────────────────────────────────────
        self.mode          = mode
        self.action_repeat = action_repeat
        self.randomize     = randomize
        self.render_mode   = render_mode
        self._home_pos = self.home_position.copy()

        # ── IK / trajectory (accept either instance or name) ──────────────
        if ik_solver is None and ik_name is not None and mode == "task_space":
            ik_solver = _build_ik(ik_name, self.model, self.data)
        if traj_gen is None and traj_name is not None:
            traj_gen = _build_traj(traj_name)

        # ── Control pipeline ──────────────────────────────────────────────
        self._pipeline = ControlPipeline(
            env=self,
            ik_solver=ik_solver,
            traj_gen=traj_gen,
            mode=mode,
            home_pos=self._home_pos,
            action_repeat=action_repeat,
        )

        # ── Reward calculator ─────────────────────────────────────────────
        self._reward_calc = RewardCalculator(self)

        # ── Gymnasium spaces ──────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32
        )
        if mode == "task_space":
            self.action_space = spaces.Box(
                low=ACTION_LOW_TASK, high=ACTION_HIGH_TASK, dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-np.ones(ACT_DIM_JOINT, dtype=np.float32),
                high=np.ones(ACT_DIM_JOINT, dtype=np.float32),
                dtype=np.float32,
            )

        # ── Viewer / RL episode trackers ──────────────────────────────────
        self._viewer       = None
        self._episode_step = 0
        self._total_reward = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Raw simulation helpers (formerly src.simulation.environment.Environment)
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_ball(self):
        """Find the ball free-joint body in the MuJoCo model."""
        try:
            self.ball_body_id = self.model.body("ball").id
            print("Found existing ball body in scene.xml")
        except KeyError:
            print("Ball will be spawned dynamically at reset")
            self.ball_body_id = None

    def _load_tested_ball_states(self, ball_cfg: dict) -> t.List[t.Dict[str, np.ndarray]]:
        """
        Parse optional curated spawn states from config.

        Expected YAML schema (under ``ball``):
            tested_initial_states:
              - position: [x, y, z]
                velocity: [vx, vy, vz]
                spin: [wx, wy, wz]      # optional

        Invalid entries are ignored. If none are valid, an empty list is
        returned and the environment falls back to range-based random spawn.
        """
        raw_states = ball_cfg.get("tested_initial_states", [])
        parsed: t.List[t.Dict[str, np.ndarray]] = []

        if not isinstance(raw_states, list):
            return parsed

        for item in raw_states:
            if not isinstance(item, dict):
                continue
            if "position" not in item or "velocity" not in item:
                continue

            try:
                position = np.asarray(item["position"], dtype=float).reshape(3)
                velocity = np.asarray(item["velocity"], dtype=float).reshape(3)
                spin_raw = item.get("spin", [0.0, 0.0, 0.0])
                spin = np.asarray(spin_raw, dtype=float).reshape(3)
            except Exception:
                continue

            parsed.append({
                "position": position,
                "velocity": velocity,
                "spin": spin,
            })

        if parsed:
            print(f"Loaded {len(parsed)} tested ball initial state(s) from config")
        return parsed

    def _sim_step(self, ctrl_commands: np.ndarray) -> t.Tuple[np.ndarray, float, bool, dict]:
        """
        Advance the simulation by exactly one MuJoCo timestep.

        This is the *low-level* interface; it does **not** go through the RL
        action pipeline.  Use it from test scripts and benchmarks.

        Parameters
        ----------
        ctrl_commands : (n_dof,) joint-position targets for the position actuators.

        Returns
        -------
        obs     : (23,) legacy observation vector — ``get_observation()``
        reward  : always ``0.0`` (shaped reward is computed in ``step()``)
        done    : ``True`` when a termination condition fires
        info    : ``dict`` with ``episode_time``, ``episode_steps``, ``done_reason``
        """
        self.data.ctrl[:self.n_dof] = ctrl_commands
        mujoco.mj_step(self.model, self.data)
        self.episode_time  += self.dt
        self.episode_steps += 1

        obs          = self.get_observation()
        done, reason = self._check_termination()
        info = {
            "episode_time":  self.episode_time,
            "episode_steps": self.episode_steps,
            "done_reason":   reason if done else None,
        }
        return obs, 0.0, done, info

    def set_robot_joints(self, positions: np.ndarray, velocities: t.Optional[np.ndarray] = None):
        """Set robot joint positions and optionally velocities."""
        self.data.qpos[:self.n_dof] = positions
        if velocities is not None:
            self.data.qvel[:self.n_dof] = velocities
        else:
            self.data.qvel[:self.n_dof] = np.zeros(self.n_dof)

    def get_robot_state(self) -> t.Dict[str, np.ndarray]:
        """Return ``{"position": (7,), "velocity": (7,)}`` joint state."""
        return {
            "position": self.data.qpos[:self.n_dof].copy(),
            "velocity": self.data.qvel[:self.n_dof].copy(),
        }

    def get_end_effector_pose(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Return paddle contact-site ``(position (3,), quaternion (4,))`` in world frame."""
        pos = self.data.site_xpos[self._paddle_contact_site_id].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._paddle_contact_site_id].copy())
        return pos, quat

    def get_ball_state(self) -> t.Dict[str, np.ndarray]:
        """Return ``{"position", "velocity", "spin"}`` for the ball.  Zeros if absent."""
        if self.ball_body_id is None:
            return {"position": np.zeros(3), "velocity": np.zeros(3), "spin": np.zeros(3)}
        idx      = self.n_dof
        position = self.data.qpos[idx:idx + 3].copy()
        velocity = self.data.qvel[idx:idx + 3].copy()   # linear
        spin     = self.data.qvel[idx + 3:idx + 6].copy()  # angular
        return {"position": position, "velocity": velocity, "spin": spin}

    def set_ball_state(self, position: np.ndarray, velocity: np.ndarray, spin: t.Optional[np.ndarray] = None):
        """
        Set ball position, linear velocity, and angular velocity (spin).

        MuJoCo free-joint qvel order is ``[vx, vy, vz, ωx, ωy, ωz]``
        (linear first, angular second).
        """
        if self.ball_body_id is None:
            print("Warning: ball body not found in simulation")
            return
        idx = self.n_dof
        self.data.qpos[idx:idx + 3]     = position
        self.data.qpos[idx + 3:idx + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[idx:idx + 3]     = velocity
        self.data.qvel[idx + 3:idx + 6] = spin if spin is not None else np.zeros(3)

    def _spawn_ball_random(self):
        """Spawn ball with random position, velocity, and spin from sim config."""
        cfg = self.ball_config
        rng = getattr(self, "np_random", np.random)
        x  = rng.uniform(*cfg["spawn_x_range"])
        y  = rng.uniform(*cfg["spawn_y_range"])
        z  = rng.uniform(*cfg["spawn_height_range"])
        vx = rng.uniform(*cfg["velocity_x_range"])
        vy = rng.uniform(*cfg["velocity_y_range"])
        vz = rng.uniform(*cfg["velocity_z_range"])
        wx = rng.uniform(*cfg["spin_x_range"])
        wy = rng.uniform(*cfg["spin_y_range"])
        wz = rng.uniform(*cfg["spin_z_range"])
        self.set_ball_state(
            np.array([x, y, z]), np.array([vx, vy, vz]), np.array([wx, wy, wz])
        )

    def _spawn_ball_from_tested_states(self):
        """Spawn ball by sampling one pre-validated state from config list."""
        if not self._tested_ball_states:
            self._spawn_ball_random()
            return

        rng = getattr(self, "np_random", np.random)
        idx = int(rng.integers(0, len(self._tested_ball_states)))
        state = self._tested_ball_states[idx]
        self.set_ball_state(
            state["position"].copy(),
            state["velocity"].copy(),
            state["spin"].copy(),
        )

    def _spawn_ball_default(self):
        """Spawn ball from opponent side, heading toward the robot."""
        self.set_ball_state(
            np.array([-1.0, 0.0, 1.5]),
            np.array([ 4.0, 0.0, -1.0]),
            np.zeros(3),
        )

    def _park_ball(self):
        """Teleport ball 10 m underground between episodes (MuJoCo has no body removal)."""
        if self.ball_body_id is None:
            return
        idx = self.n_dof
        self.data.qpos[idx:idx + 3]     = np.array([0.0, 0.0, -10.0])
        self.data.qpos[idx + 3:idx + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[idx:idx + 6]     = np.zeros(6)

    def _check_termination(self) -> t.Tuple[bool, t.Optional[str]]:
        """Check all episode termination conditions."""
        ball = self.get_ball_state()

        if ball["position"][2] < -5.0:
            return True, "ball_parked"

        vel_norm  = float(np.linalg.norm(ball["velocity"]))
        near_floor = ball["position"][2] < 0.85
        if vel_norm < self.min_velocity_threshold and self.episode_steps > 50 and near_floor:
            self._park_ball()
            return True, "ball_stopped"

        # End immediately on first floor touch so the ball is "destroyed"
        # right when it hits the ground rather than rolling away.
        ground_z_hit = max(self.ground_z_threshold, self.ball_radius + 1e-3)
        if self._ball_touches_floor() or ball["position"][2] < ground_z_hit:
            self._park_ball()
            return True, "ball_hit_ground"

        if float(np.linalg.norm(ball["position"][:2])) > self.out_of_bounds_dist:
            self._park_ball()
            return True, "ball_out_of_bounds"

        if self.episode_time >= self.max_episode_time:
            self._park_ball()
            return True, "max_time_exceeded"

        return False, None

    def _ball_touches_floor(self) -> bool:
        """Return True when MuJoCo reports a ball↔floor contact pair."""
        try:
            for ci in range(self.data.ncon):
                c = self.data.contact[ci]
                g1 = self.model.geom(int(c.geom1)).name
                g2 = self.model.geom(int(c.geom2)).name
                pair = {g1, g2}
                if "ball_geom" in pair and pair & self._floor_geom_names:
                    return True
        except Exception:
            pass
        return False

    def get_observation(self) -> np.ndarray:
        """23-D legacy observation: ``[ball(9), joint_pos(7), joint_vel(7)]``."""
        ball  = self.get_ball_state()
        robot = self.get_robot_state()
        return np.concatenate([
            ball["position"],    # 3
            ball["velocity"],    # 3
            ball["spin"],        # 3
            robot["position"],   # 7
            robot["velocity"],   # 7
        ])

    def get_paddle_normal(self) -> np.ndarray:
        """Return the unit normal of the paddle face (local Z → world frame)."""
        mat     = self.data.site_xmat[self._paddle_normal_site_id].reshape(3, 3)
        normal  = mat[:, 2].copy()
        norm    = float(np.linalg.norm(normal))
        return normal / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    def get_rl_observation(self) -> np.ndarray:
        """19-D RL observation: ``[ball(9), joint_pos(7), paddle_normal(3)]``."""
        ball   = self.get_ball_state()
        robot  = self.get_robot_state()
        normal = self.get_paddle_normal()
        return np.concatenate([
            ball["position"],    # 3
            ball["velocity"],    # 3
            ball["spin"],        # 3
            robot["position"],   # 7
            normal,              # 3
        ]).astype(np.float32)

    def get_simulation_time(self) -> float:
        """Return current MuJoCo simulation time (seconds)."""
        return float(self.data.time)

    def get_contact_forces(self) -> t.Dict[str, np.ndarray]:
        """Return ``{geom1-geom2: force_6d}`` contact-force dictionary."""
        contacts = {}
        for i in range(self.data.ncon):
            c     = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            g1 = self.model.geom(c.geom1).name
            g2 = self.model.geom(c.geom2).name
            contacts[f"{g1}-{g2}"] = force
        return contacts

    def apply_joint_torques(self, torques: np.ndarray):
        """Write joint torques to the motor actuators (``ctrl[n_dof:2*n_dof]``)."""
        self.data.ctrl[:self.n_dof]                = 0.0
        self.data.ctrl[self.n_dof:2 * self.n_dof] = torques

    def set_home_position(self, position: np.ndarray):
        """Override the home joint configuration used at episode start."""
        self.home_position = position.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None) -> t.Tuple[np.ndarray, dict]:
        """
        Reset to a fresh episode.

        The ``options`` dict may contain ``{"randomize": bool}`` to override
        the instance-level ``self.randomize`` setting for this one reset.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        randomize = self.randomize
        if options is not None:
            randomize = options.get("randomize", randomize)

        mujoco.mj_resetData(self.model, self.data)
        self._park_ball()
        self.set_robot_joints(self._home_pos, np.zeros(self.n_dof))

        if randomize:
            self._spawn_ball_from_tested_states()
        else:
            self._spawn_ball_default()

        self.episode_time  = 0.0
        self.episode_steps = 0

        mujoco.mj_forward(self.model, self.data)

        self._pipeline.reset()
        self._reward_calc.reset()
        self._episode_step = 0
        self._total_reward = 0.0

        return self.get_rl_observation(), {}

    def step(self, action: np.ndarray) -> t.Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply one RL action for ``action_repeat`` simulation steps.

        Parameters
        ----------
        action : np.ndarray
            ``task_space``: (10,) task-space action.
            ``joint_space``: (7,) normalised joint targets ∈ [-1, 1].

        Returns
        -------
        obs, reward, terminated, truncated, info  (standard Gymnasium 5-tuple)
        """
        current_q = self.get_robot_state()["position"]
        self._pipeline.plan(action, current_q)

        ep_reward   = 0.0
        terminated  = False
        truncated   = False
        done_reason = None

        _INSTABILITY_PENALTY = -20.0
        _QACC_LIMIT          = 1e5
        _REWARD_CLIP         = 10.0

        for inner in range(self.action_repeat):
            ctrl               = self._pipeline.get_ctrl(inner)
            _, _, done, iinfo  = self._sim_step(ctrl)

            # ── Instability guard ────────────────────────────────────────
            qacc = self.data.qacc[:self.n_dof]
            qpos = self.data.qpos[:self.n_dof]
            if (not np.all(np.isfinite(qpos))
                    or not np.all(np.isfinite(qacc))
                    or np.any(np.abs(qacc) > _QACC_LIMIT)):
                ep_reward  += _INSTABILITY_PENALTY
                terminated  = True
                done_reason = "sim_instability"
                break

            ball      = self.get_ball_state()
            paddle, _ = self.get_end_effector_pose()
            paddle_normal = self.get_paddle_normal()
            robot     = self.get_robot_state()

            step_reward = self._reward_calc.update(
                ball_pos    = ball["position"],
                ball_vel    = ball["velocity"],
                paddle_pos  = paddle,
                paddle_normal = paddle_normal,
                joint_pos   = robot["position"],
                joint_vel   = robot["velocity"],
                done_reason = iinfo.get("done_reason"),
                step        = self._episode_step,
            )
            ep_reward          += float(np.clip(step_reward, -_REWARD_CLIP, _REWARD_CLIP))
            self._episode_step += 1

            if done:
                done_reason = iinfo.get("done_reason", "unknown")
                terminated  = done_reason != "max_time_exceeded"
                truncated   = done_reason == "max_time_exceeded"
                break

            if self.render_mode == "human":
                self.render()

        self._total_reward += ep_reward
        obs = np.clip(self.get_rl_observation(), OBS_LOW, OBS_HIGH)

        info_out = {
            "done_reason":      done_reason,
            "episode_reward":   self._total_reward,
            "hit_detected":     self._reward_calc.hit_detected,
            "hit_after_bounce": self._reward_calc.hit_after_bounce,
            "hit_with_other_side": self._reward_calc.hit_with_other_side,
            "net_contacted":    self._reward_calc.net_contacted,
            "over_net":         self._reward_calc.over_net,
            "landed_far_side":  self._reward_calc.landed_far_side,
            "missed":           self._reward_calc.missed,
        }
        return obs, ep_reward, terminated, truncated, info_out

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal:  np.ndarray, info: dict) -> float:
        """
        Stub for Gymnasium-Robotics / HER compatibility.

        Returns the ``episode_reward`` stored in ``info`` when available,
        otherwise 0.0.
        """
        return float(info.get("episode_reward", 0.0))

# ── Factory helper ────────────────────────────────────────────────────────────

def make_env(mode: str = "task_space", ik_name: t.Optional[str] = "NumericalIKSolver", traj_name: t.Optional[str] = "MinimumJerkTrajectory", randomize: bool = True, action_repeat: int = 20, render_mode: t.Optional[str] = None, scene_xml: str = "assets/scene.xml", sim_cfg_path: str = "config/simulation.yaml", robot_cfg_path: str = "config/robot.yaml") -> Environment:
    """
    Build and return a fully initialised ``TableTennisEnv``.

    Example
    -------
    env = make_env(mode="task_space", ik_name="NumericalIKSolver",
                   traj_name="MinimumJerkTrajectory")
    obs, _ = env.reset()
    """
    return Environment(mode=mode, ik_name=ik_name if mode == "task_space" else None, traj_name=traj_name, action_repeat=action_repeat, randomize=randomize, render_mode=render_mode, scene_xml=scene_xml, sim_cfg_path=sim_cfg_path, robot_cfg_path=robot_cfg_path)
