"""
src.simulation.environment — compatibility shim
================================================

The ``Environment`` class previously defined here has been merged into
``src.rl.gym_env.TableTennisEnv``, which is a proper ``gymnasium.Env``
subclass with all simulation logic built in.

This module re-exports ``TableTennisEnv`` under the legacy name
``Environment`` so that any external code using::

    from src.simulation.environment import Environment

continues to work without modification.

**Do not add new code here.**  Use ``src.rl.gym_env.TableTennisEnv`` directly
in all new code.
"""

from src.rl.gym_env import Environment as Environment  # noqa: F401

__all__ = ["Environment"]
