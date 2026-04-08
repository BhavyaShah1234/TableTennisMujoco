"""
Utility functions for the table tennis robot project.

Contains:
- Configuration loading
- Mathematical utilities (rotations, transformations)
- Common helper functions
"""

import time
import yaml as y
import numpy as np
import typing as t
import pathlib as p


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ═══════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> t.Dict[str, t.Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file (relative or absolute)
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    path = p.Path(config_path)
    with open(path, 'r') as f:
        config = y.safe_load(f)
    return config

def get_project_root() -> p.Path:
    """
    Get the project root directory.
    
    Returns:
        p.Path object pointing to project root
    """
    # Assumes this file is in src/utils/
    return p.Path(__file__).parent.parent.parent

# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (XYZ convention) to rotation matrix.
    
    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def rotation_matrix_to_euler(R: np.ndarray) -> t.Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (XYZ convention).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return roll, pitch, yaw

def quaternion_to_euler(quat: np.ndarray) -> t.Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (XYZ convention).
    
    Args:
        quat: Quaternion [w, x, y, z]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)
        
    Returns:
        Quaternion [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def clip_to_limits(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clip values to specified limits."""
    return np.clip(value, lower, upper)

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


# ═══════════════════Any════════════════════════════════════════════════
# TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.time_elapsed = 0.0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            return 0.0

        self.time_elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.time_elapsed
    
    def elapsed(self) -> float:
        """Get current elapsed time without stopping."""
        if self.start_time is None:
            return self.time_elapsed
        return time.perf_counter() - self.start_time

def validate_array_shape(arr: np.ndarray, expected_shape: t.Tuple, name: str = "array"):
    """Validate that an array has the expected shape."""
    if arr.shape != expected_shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}")

def validate_finite(arr: np.ndarray, name: str = "array"):
    """Validate that all array elements are finite (no NaN or inf)."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")
