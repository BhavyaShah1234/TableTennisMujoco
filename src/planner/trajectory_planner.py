"""
Trajectory planning and generation for smooth robot motion.

This module provides trajectory generators that create smooth paths in joint space.
All generators inherit from a base class that ensures consistent interface.
"""

import abc
import numpy as np
import typing as t
from scipy.interpolate import CubicSpline, make_interp_spline

# ═══════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════

class BaseTrajectoryGenerator(abc.ABC):
    """
    Abstract base class for trajectory generators.
    
    All trajectory generators must implement:
    - generate_trajectory(): Create smooth trajectory between waypoints
    - evaluate_at_time(): Get trajectory state at specific time
    """
    
    @abc.abstractmethod
    def generate_trajectory(self, waypoints: np.ndarray, times: np.ndarray, dt: float) -> t.List[t.Dict[str, np.ndarray]]:
        """
        Generate complete trajectory through waypoints.
        
        Args:
            waypoints: (N, n_dof) array of waypoints to pass through
            times: (N,) array of times for each waypoint
            dt: Timestep for trajectory discretization
            
        Returns:
            List of trajectory points, each containing:
                - 'time': scalar time
                - 'position': (n_dof,) joint positions
                - 'velocity': (n_dof,) joint velocities
                - 'acceleration': (n_dof,) joint accelerations
        """
        pass
    
    @abc.abstractmethod
    def evaluate_at_time(self, t: float) -> t.Dict[str, np.ndarray]:
        """
        Evaluate trajectory at a specific time.
        
        Args:
            t: Time at which to evaluate trajectory
            
        Returns:
            t.Dictionary with 'time', 'position', 'velocity', 'acceleration'
        """
        pass

# ═══════════════════════════════════════════════════════════════════
# MINIMUM JERK TRAJECTORY (5TH ORDER POLYNOMIAL)
# ═══════════════════════════════════════════════════════════════════

class MinimumJerkTrajectory(BaseTrajectoryGenerator):
    """
    Minimum jerk trajectory using 5th-order polynomial.
    
    Generates the smoothest possible motion (minimizes jerk = d³x/dt³).
    
    The trajectory follows:
        s(τ) = 10τ³ - 15τ⁴ + 6τ⁵  where τ = t/T ∈ [0,1]
    
    This creates very human-like, natural motion that's ideal for table tennis
    where smooth acceleration is critical for accurate ball striking.
    """
    
    def __init__(self):
        self.q_start = None
        self.q_goal = None
        self.T = None
    
    def generate_trajectory(self, waypoints: np.ndarray, times: np.ndarray, dt: float) -> t.List[t.Dict[str, np.ndarray]]:
        """
        Generate minimum jerk trajectory.
        
        For minimum jerk, we only use first and last waypoints (point-to-point).
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        self.q_start = waypoints[0]
        self.q_goal = waypoints[-1]
        self.T = times[-1] - times[0]
        
        # Generate trajectory points
        trajectory = []
        num_steps = int(self.T / dt) + 1
        
        for i in range(num_steps):
            t = i * dt
            state = self.evaluate_at_time(t)
            trajectory.append(state)
        
        return trajectory
    
    def evaluate_at_time(self, t: float) -> t.Dict[str, np.ndarray]:
        """Evaluate minimum jerk trajectory at time t."""
        if self.q_start is None or self.q_goal is None:
            raise RuntimeError("Must call generate_trajectory first")
        
        # Normalized time τ ∈ [0, 1]
        tau = np.clip(t / self.T, 0.0, 1.0)
        
        # Minimum jerk polynomial and derivatives
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        sd = (30*tau**2 - 60*tau**3 + 30*tau**4) / self.T
        sdd = (60*tau - 180*tau**2 + 120*tau**3) / (self.T**2)
        
        # Interpolate between start and goal
        q = self.q_start + (self.q_goal - self.q_start) * s
        qd = (self.q_goal - self.q_start) * sd
        qdd = (self.q_goal - self.q_start) * sdd
        
        return {
            'time': t,
            'position': q,
            'velocity': qd,
            'acceleration': qdd
        }

# ═══════════════════════════════════════════════════════════════════
# CUBIC SPLINE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════

class CubicSplineTrajectory(BaseTrajectoryGenerator):
    """
    Cubic spline trajectory through multiple waypoints.
    
    Uses piecewise cubic polynomials to create smooth path through all waypoints.
    Ensures C² continuity (position, velocity, acceleration are continuous).
    
    This is particularly useful when you have multiple waypoints from a path planner
    (like A* or RRT) and need to create a smooth trajectory through all of them.
    Unlike minimum jerk which only connects two points, cubic splines handle
    arbitrary numbers of waypoints while maintaining smoothness.
    """
    
    def __init__(self, boundary_condition: str = 'clamped'):
        """
        Initialize cubic spline trajectory generator.
        
        Args:
            boundary_condition: 'clamped' means zero velocity at endpoints,
                              'natural' means zero acceleration at endpoints
        """
        self.boundary_condition = boundary_condition
        self.spline = None
        self.n_dof = None
    
    def generate_trajectory(self, waypoints: np.ndarray, times: np.ndarray, dt: float) -> t.List[t.Dict[str, np.ndarray]]:
        """Generate cubic spline trajectory through waypoints."""
        if len(waypoints) != len(times):
            raise ValueError("Number of waypoints must match number of times")
        
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        self.n_dof = waypoints.shape[1]
        
        # Create cubic spline for each joint
        # scipy's CubicSpline automatically handles the interpolation
        self.spline = CubicSpline(times, waypoints, bc_type=self.boundary_condition)
        
        # Generate trajectory points
        trajectory = []
        t_start = times[0]
        t_end = times[-1]
        num_steps = int((t_end - t_start) / dt) + 1
        
        for i in range(num_steps):
            t = t_start + i * dt
            if t > t_end:
                t = t_end
            state = self.evaluate_at_time(t)
            trajectory.append(state)
        
        return trajectory
    
    def evaluate_at_time(self, t: float) -> t.Dict[str, np.ndarray]:
        """Evaluate cubic spline at time t."""
        if self.spline is None:
            raise RuntimeError("Must call generate_trajectory first")
        
        # CubicSpline can compute derivatives automatically
        q = self.spline(t)          # Position
        qd = self.spline(t, 1)      # First derivative (velocity)
        qdd = self.spline(t, 2)     # Second derivative (acceleration)
        
        return {
            'time': t,
            'position': q,
            'velocity': qd,
            'acceleration': qdd
        }


# ═══════════════════════════════════════════════════════════════════
# B-SPLINE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════

class BSplineTrajectory(BaseTrajectoryGenerator):
    """
    B-spline trajectory for smooth motion with local control.
    
    B-splines provide local control, meaning that moving one control point only 
    affects the nearby curve, not the entire trajectory. This is different from
    cubic splines where changing one waypoint can affect the entire path.
    
    The tradeoff is that B-splines do not pass through the control points exactly,
    they only approximate them. This is actually beneficial in many cases because
    it provides smoother motion without sharp corners at waypoints.
    
    B-splines are the industry standard in CAD, animation, and robotics for
    creating smooth, controllable curves.
    """
    
    def __init__(self, degree: int = 3):
        """
        Initialize B-spline trajectory generator.
        
        Args:
            degree: Degree of spline (3 = cubic, 5 = quintic)
                   Higher degree = smoother but requires more waypoints
        """
        self.degree = degree
        self.spline = None
        self.t_start = None
        self.t_end = None
    
    def generate_trajectory(self, waypoints: np.ndarray, times: np.ndarray, dt: float) -> t.List[t.Dict[str, np.ndarray]]:
        """Generate B-spline trajectory."""
        if len(waypoints) < self.degree + 1:
            raise ValueError(f"Need at least {self.degree + 1} waypoints for degree {self.degree}")
        
        n_points = len(waypoints)
        self.t_start = times[0]
        self.t_end = times[-1]
        
        # Create uniform parameter space [0, 1]
        # This is the internal parameter that the B-spline uses
        t_params = np.linspace(0, 1, n_points)
        
        # Create B-spline using scipy's interpolation
        # This creates a smooth curve that approximates the waypoints
        self.spline = make_interp_spline(t_params, waypoints, k=self.degree)
        
        # Generate trajectory points
        trajectory = []
        num_steps = int((self.t_end - self.t_start) / dt) + 1
        
        for i in range(num_steps):
            t = self.t_start + i * dt
            if t > self.t_end:
                t = self.t_end
            
            # Map actual time to parameter space [0, 1]
            tau = (t - self.t_start) / (self.t_end - self.t_start)
            state = self._evaluate_at_param(tau, t)
            trajectory.append(state)
        
        return trajectory
    
    def evaluate_at_time(self, t: float) -> t.Dict[str, np.ndarray]:
        """Evaluate B-spline at time t."""
        if self.spline is None:
            raise RuntimeError("Must call generate_trajectory first")
        
        # Map to parameter space [0, 1]
        tau = (t - self.t_start) / (self.t_end - self.t_start)
        tau = np.clip(tau, 0.0, 1.0)
        
        return self._evaluate_at_param(tau, t)
    
    def _evaluate_at_param(self, tau: float, t: float) -> t.Dict[str, np.ndarray]:
        """
        Evaluate B-spline at parameter tau.
        
        This is an internal helper that handles the parameter space conversion.
        The B-spline works in parameter space [0,1], but we need to convert
        derivatives back to real time.
        """
        q = self.spline(tau)
        # Chain rule: dq/dt = dq/dτ * dτ/dt = dq/dτ / (t_end - t_start)
        qd = self.spline(tau, 1) / (self.t_end - self.t_start)
        qdd = self.spline(tau, 2) / ((self.t_end - self.t_start)**2)
        
        return {
            'time': t,
            'position': q,
            'velocity': qd,
            'acceleration': qdd
        }


# ═══════════════════════════════════════════════════════════════════
# TRAPEZOIDAL VELOCITY PROFILE
# ═══════════════════════════════════════════════════════════════════

class TrapezoidalVelocityTrajectory(BaseTrajectoryGenerator):
    """
    Trapezoidal velocity profile (bang-bang control).
    
    Three phases:
    1. Constant acceleration (bang) - accelerate as fast as possible
    2. Constant velocity (coast) - maintain maximum velocity
    3. Constant deceleration (bang) - decelerate as fast as possible
    
    The velocity profile looks like a trapezoid when plotted over time, hence the name.
    
    This is the time-optimal trajectory for given velocity and acceleration limits.
    However, it has discontinuous acceleration (infinite jerk at phase transitions),
    which can cause vibrations and wear on mechanical systems.
    
    Industrial robots often use this for pick-and-place tasks where speed is more
    important than smoothness. For table tennis, minimum jerk is usually better
    because the smooth acceleration is critical for accurate ball striking.
    """
    
    def __init__(self, max_velocity: float, max_acceleration: float):
        """
        Initialize trapezoidal velocity trajectory.
        
        Args:
            max_velocity: Maximum velocity (rad/s or m/s)
            max_acceleration: Maximum acceleration (rad/s² or m/s²)
        """
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.q_start = None
        self.q_goal = None
        self.T = None
        self.t_accel = None
        self.t_const = None
        self.t_decel = None
    
    def generate_trajectory(self, waypoints: np.ndarray, times: np.ndarray, dt: float) -> t.List[t.Dict[str, np.ndarray]]:
        """
        Generate trapezoidal velocity trajectory.
        
        Only uses first and last waypoints (point-to-point).
        The duration is computed from the velocity and acceleration limits,
        not from the times array.
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        self.q_start = waypoints[0]
        self.q_goal = waypoints[-1]
        
        # Calculate the trajectory duration based on physics
        # For each joint, compute how long it takes to reach the goal
        # Then use the longest duration (the bottleneck joint)
        max_duration = 0.0
        
        for i in range(len(self.q_start)):
            distance = abs(self.q_goal[i] - self.q_start[i])
            
            # Time to reach max velocity from rest
            t_a = self.v_max / self.a_max
            
            # Distance covered during acceleration and deceleration
            d_accel = 0.5 * self.a_max * t_a**2
            
            if 2 * d_accel >= distance:
                # Triangular profile (never reaches max velocity)
                # This happens when the distance is too short
                t_a = np.sqrt(distance / self.a_max)
                t_c = 0.0
            else:
                # Trapezoidal profile (reaches max velocity)
                t_c = (distance - 2 * d_accel) / self.v_max
            
            total_time = 2 * t_a + t_c
            max_duration = max(max_duration, total_time)
        
        self.T = max_duration
        self.t_accel = self.v_max / self.a_max
        
        # Recalculate constant velocity phase duration
        distance_total = np.linalg.norm(self.q_goal - self.q_start)
        distance_accel = 0.5 * self.a_max * self.t_accel**2
        
        if 2 * distance_accel < distance_total:
            self.t_const = (distance_total - 2 * distance_accel) / self.v_max
        else:
            self.t_const = 0.0
            self.t_accel = np.sqrt(distance_total / self.a_max)
        
        self.t_decel = self.t_accel
        
        # Generate trajectory
        trajectory = []
        num_steps = int(self.T / dt) + 1
        
        for i in range(num_steps):
            t = i * dt
            if t > self.T:
                t = self.T
            state = self.evaluate_at_time(t)
            trajectory.append(state)
        
        return trajectory
    
    def evaluate_at_time(self, t: float) -> t.Dict[str, np.ndarray]:
        """Evaluate trapezoidal trajectory at time t."""
        if self.q_start is None:
            raise RuntimeError("Must call generate_trajectory first")
        
        t = np.clip(t, 0.0, self.T)
        
        # Direction vector (normalized)
        direction = (self.q_goal - self.q_start)
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
        
        # Determine which phase we're in and calculate motion accordingly
        if t <= self.t_accel:
            # Phase 1: Acceleration
            # Use kinematic equation: s = 0.5 * a * t²
            s = 0.5 * self.a_max * t**2
            sd = self.a_max * t
            sdd = self.a_max
        elif t <= self.t_accel + self.t_const:
            # Phase 2: Constant velocity
            # Position continues from end of acceleration phase
            t_rel = t - self.t_accel
            s = 0.5 * self.a_max * self.t_accel**2 + self.v_max * t_rel
            sd = self.v_max
            sdd = 0.0
        else:
            # Phase 3: Deceleration
            # Symmetrical to acceleration phase but going backwards
            t_rel = t - self.t_accel - self.t_const
            s = (0.5 * self.a_max * self.t_accel**2 + 
                 self.v_max * self.t_const +
                 self.v_max * t_rel - 0.5 * self.a_max * t_rel**2)
            sd = self.v_max - self.a_max * t_rel
            sdd = -self.a_max
        
        # Convert scalar motion to joint space using direction vector
        q = self.q_start + direction * s
        qd = direction * sd
        qdd = direction * sdd
        
        return {'time': t, 'position': q, 'velocity': qd, 'acceleration': qdd}
