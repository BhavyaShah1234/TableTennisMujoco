"""
Base controller classes and implementations for robot control.

This module provides an abstract base class for controllers and concrete
implementations of various control strategies (PD, PID, Computed Torque, etc.).
"""

import abc
import numpy as np
import mujoco as m
import typing as t

# ═══════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════

class BaseController(abc.ABC):
    """
    Abstract base class for all robot controllers.
    
    All controllers must implement:
    - compute_control(): Calculate control torques
    - reset(): Reset internal state
    """
    
    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int):
        """
        Initialize base controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            n_dof: Number of degrees of freedom (joints)
        """
        self.model = model
        self.data = data
        self.n_dof = n_dof

    @abc.abstractmethod
    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute control torques.
        
        Args:
            desired_state: Dictionary with 'position', 'velocity', 'acceleration'
            actual_state: Dictionary with 'position', 'velocity'
        
        Returns:
            (n_dof,) array of control torques
        """
        pass
    
    @abc.abstractmethod
    def reset(self):
        """Reset controller internal state."""
        pass
    
    def get_joint_states(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Get current joint positions and velocities from MuJoCo data."""
        position = self.data.qpos[:self.n_dof].copy()
        velocity = self.data.qvel[:self.n_dof].copy()
        return position, velocity


# ═══════════════════════════════════════════════════════════════════
# PD POSITION CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class PDController(BaseController):
    """
    Proportional-Derivative position controller.
    
    Control law: τ = Kp * (q_desired - q_actual) + Kd * (q̇_desired - q̇_actual)
    """
    
    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int, kp: np.ndarray, kd: np.ndarray):
        super().__init__(model, data, n_dof)
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        
        if self.kp.shape != (n_dof,) or self.kd.shape != (n_dof,):
            raise ValueError(f"Gains must have shape ({n_dof},)")

    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        q_desired = desired_state['position']
        q_actual = actual_state['position']
        qd_desired = desired_state.get('velocity', np.zeros(self.n_dof))
        qd_actual = actual_state['velocity']
        
        position_error = q_desired - q_actual
        velocity_error = qd_desired - qd_actual
        
        torque = self.kp * position_error + self.kd * velocity_error
        return torque
    
    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════
# PID CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class PIDController(BaseController):
    """
    Proportional-Integral-Derivative controller.
    
    Control law: τ = Kp*e + Kd*ė + Ki*∫e dt
    """

    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int, kp: np.ndarray, kd: np.ndarray, ki: np.ndarray, max_integral: float = 10.0, dt: float = 0.001):
        super().__init__(model, data, n_dof)
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.ki = np.array(ki)
        self.max_integral = max_integral
        self.dt = dt
        self.integral_error = np.zeros(n_dof)

        if self.kp.shape != (n_dof,) or self.kd.shape != (n_dof,) or self.ki.shape != (n_dof,):
            raise ValueError(f"All gains must have shape ({n_dof},)")
    
    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        q_desired = desired_state['position']
        q_actual = actual_state['position']
        qd_desired = desired_state.get('velocity', np.zeros(self.n_dof))
        qd_actual = actual_state['velocity']
        
        position_error = q_desired - q_actual
        velocity_error = qd_desired - qd_actual
        
        # Update integral with anti-windup
        self.integral_error += position_error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)
        torque = (self.kp * position_error) + (self.kd * velocity_error) + (self.ki * self.integral_error)
        return torque

    def reset(self):
        self.integral_error = np.zeros(self.n_dof)

# ═══════════════════════════════════════════════════════════════════
# COMPUTED TORQUE CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class ComputedTorqueController(BaseController):
    """
    Model-based computed torque controller with full dynamics compensation.
    
    Control law:
        τ = M(q)*[q̈_desired + Kp*e + Kd*ė] + C(q,q̇) + G(q)
    
    This is the most accurate controller when the model is correct.
    It compensates for:
    - Inertia effects (M(q))
    - Coriolis and centrifugal forces (C(q,q̇))
    - Gravity (G(q))
    """
    
    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int, kp: np.ndarray, kd: np.ndarray, use_gravity_compensation: bool = True, use_coriolis_compensation: bool = True):
        super().__init__(model, data, n_dof)
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.use_gravity_comp = use_gravity_compensation
        self.use_coriolis_comp = use_coriolis_compensation
        
        if self.kp.shape != (n_dof,) or self.kd.shape != (n_dof,):
            raise ValueError(f"Gains must have shape ({n_dof},)")
    
    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        # Extract states
        q_desired = desired_state['position']
        qd_desired = desired_state['velocity']
        qdd_desired = desired_state.get('acceleration', np.zeros(self.n_dof))
        
        q_actual = actual_state['position']
        qd_actual = actual_state['velocity']
        
        # Compute errors
        position_error = q_desired - q_actual
        velocity_error = qd_desired - qd_actual
        
        # Desired acceleration with feedback correction
        qdd_command = qdd_desired + self.kp * position_error + self.kd * velocity_error
        
        # Get inertia matrix M(q)
        M = self._get_inertia_matrix()
        
        # Feedforward term: M(q) * q̈_command
        feedforward = M @ qdd_command
        
        # Get bias forces (Coriolis + gravity)
        bias = self._get_bias_forces()
        
        # Total torque
        torque = feedforward + bias
        
        return torque
    
    def _get_inertia_matrix(self) -> np.ndarray:
        """
        Extract inertia matrix M(q) from MuJoCo.
        
        MuJoCo stores the mass matrix in a special compact format (qM).
        We use mj_fullM to expand it into the full matrix form.
        """
        M_full = np.zeros((self.model.nv, self.model.nv))
        m.mj_fullM(self.model, M_full, self.data.qM)
        
        # Extract only the robot joints (first n_dof x n_dof block)
        M = M_full[:self.n_dof, :self.n_dof]
        
        return M
    
    def _get_bias_forces(self) -> np.ndarray:
        """
        Get bias forces (Coriolis + centrifugal + gravity).
        
        MuJoCo provides this in data.qfrc_bias, which contains:
        C(q,q̇)*q̇ + G(q)
        """
        bias = self.data.qfrc_bias[:self.n_dof].copy()
        return bias
    
    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════
# VELOCITY CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class VelocityController(BaseController):
    """
    Simple velocity controller.
    
    Control law: τ = Kv * (q̇_desired - q̇_actual)
    
    Useful for velocity tracking tasks but doesn't account for dynamics.
    """
    
    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int, kv: np.ndarray):
        super().__init__(model, data, n_dof)
        self.kv = np.array(kv)
        
        if self.kv.shape != (n_dof,):
            raise ValueError(f"Gains must have shape ({n_dof},)")
    
    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        qd_desired = desired_state['velocity']
        qd_actual = actual_state['velocity']
        
        velocity_error = qd_desired - qd_actual
        torque = self.kv * velocity_error
        
        return torque
    
    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════
# IMPEDANCE CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class ImpedanceController(BaseController):
    """
    Impedance controller for compliant interaction.
    
    Control law:
        τ = K_d*(q_desired - q) + D_d*(q̇_desired - q̇)
    
    Creates spring-damper behavior, useful for tasks requiring force control
    or compliant behavior (like contact with environment).
    """
    
    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int, desired_stiffness: np.ndarray, desired_damping: np.ndarray, desired_inertia: t.Optional[np.ndarray] = None):
        super().__init__(model, data, n_dof)
        self.K_d = np.diag(desired_stiffness)
        self.D_d = np.diag(desired_damping)
        
        if desired_inertia is not None:
            self.M_d = np.diag(desired_inertia)
        else:
            self.M_d = None
    
    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        q_desired = desired_state['position']
        qd_desired = desired_state['velocity']
        
        q_actual = actual_state['position']
        qd_actual = actual_state['velocity']
        
        # Compute errors
        position_error = q_desired - q_actual
        velocity_error = qd_desired - qd_actual
        
        # Spring-damper behavior
        torque = self.K_d @ position_error + self.D_d @ velocity_error
        
        return torque
    
    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════
# DIRECT TORQUE CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class DirectTorqueController(BaseController):
    """
    Pass-through controller that applies torques directly.
    
    Useful for:
    - Testing
    - Learned controllers (RL policies that output torques)
    - Manual control
    """
    
    def __init__(self, model: m.MjModel, data: m.MjData, n_dof: int):
        super().__init__(model, data, n_dof)

    def compute_control(self, desired_state: t.Dict[str, np.ndarray], actual_state: t.Dict[str, np.ndarray]) -> np.ndarray:
        """Pass through desired torques."""
        return desired_state['torque']

    def reset(self):
        pass
