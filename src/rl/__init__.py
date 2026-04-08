"""
RL training subpackage for table tennis robot.

``TableTennisEnv`` is registered with Gymnasium as ``"TableTennis-v0"`` at
import time so that ``gymnasium.make("TableTennis-v0")`` works out of the box
(provided the project root is on ``sys.path``).
"""
from .control_pipeline import ControlPipeline
from .reward import RewardCalculator
from .gym_env import Environment, make_env

# Backward-compatible alias
Environment = Environment

__all__ = ["ControlPipeline", "RewardCalculator", "Environment", "Environment", "make_env"]

# ── Gymnasium registration ────────────────────────────────────────────────────
import gymnasium as _gym

_gym.register(
    id="TableTennis-v0",
    entry_point="src.rl.gym_env:TableTennisEnv",
    max_episode_steps=500,
    kwargs={
        "mode":      "task_space",
        "ik_name":   "NumericalIKSolver",
        "traj_name": "MinimumJerkTrajectory",
        "randomize": True,
    },
)
