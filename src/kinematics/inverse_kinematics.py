"""
Inverse kinematics solvers for converting task-space goals to joint angles.

This module provides IK solvers that convert Cartesian end-effector positions
and orientations into joint angle configurations.

The critical problem IK solves:
    Policy outputs: [x, y, z] (task-space) → Need: [θ1, ..., θ7] (joint-space)

Without IK, we cannot bridge the gap between high-level task goals and
low-level joint control. This is THE missing piece that connects your policy's
spatial goals to the trajectory generator's joint-space requirements.
"""

import abc
import numpy as np
import mujoco as m
import typing as t
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════

class BaseIKSolver(abc.ABC):
    """
    Abstract base class for inverse kinematics solvers.
    
    All IK solvers must implement the solve() method which converts a desired
    end-effector pose (position and optionally orientation) into the joint
    angles that achieve that pose.
    
    IK is fundamentally challenging because:
    1. Multiple solutions may exist (redundancy) - a 7-DOF arm can reach the
       same point in many different configurations
    2. No solution may exist (unreachable workspace) - the target might be
       too far away or blocked by obstacles
    3. The problem is nonlinear - the relationship between joint angles and
       end-effector position involves trigonometric functions and is not
       easily invertible
    """
    
    def __init__(
        self,
        model: m.MjModel,
        data: m.MjData,
        end_effector_body: str,
        end_effector_site: t.Optional[str] = None,
        end_effector_normal_site: t.Optional[str] = None,
    ):
        """
        Initialize IK solver.
        
        Args:
            model: MuJoCo model containing robot definition
            data: MuJoCo data for state and computation
            end_effector_body: Name of end-effector body (e.g., "paddle")
                             This is the body whose position we want to control
        """
        self.model = model
        self.data = data
        self.ee_body_name = end_effector_body
        self.ee_body_id = model.body(end_effector_body).id
        self.ee_site_name = end_effector_site
        self.ee_site_id = None if end_effector_site is None else model.site(end_effector_site).id
        self.ee_normal_site_name = end_effector_normal_site
        self.ee_normal_site_id = None if end_effector_normal_site is None else model.site(end_effector_normal_site).id

    @abc.abstractmethod
    def solve(self, target_position: np.ndarray, target_orientation: t.Optional[np.ndarray] = None, initial_guess: t.Optional[np.ndarray] = None, target_normal: t.Optional[np.ndarray] = None) -> t.Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics.
        
        This is the core method that every IK solver must implement. It takes
        a desired end-effector pose and returns the joint angles that achieve
        that pose (or the best approximation if exact solution doesn't exist).
        
        Args:
            target_position: (3,) desired [x, y, z] position in world frame
            target_orientation: (3,) desired [rx, ry, rz] Euler angles or (4,) quaternion
                              If None, only position is constrained (unless target_normal set)
            initial_guess: (n_dof,) starting joint configuration (optional)
                         A good initial guess can significantly improve convergence
            target_normal: (3,) desired paddle face normal direction in world frame.
                           Aligns the blade face normal (body -Y axis) with this
                           direction; twist about the normal is left unconstrained.
        
        Returns:
            Tuple of:
                - q_solution: (n_dof,) joint angles that achieve the target
                - success: True if IK converged to a valid solution
        """
        pass
    
    def forward_kinematics(self, q: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics (joint angles → end-effector pose).

        IMPORTANT: this method temporarily modifies data.qpos so that MuJoCo
        can propagate the kinematic chain to the end-effector body (which
        includes fr3_link8 and the paddle as fixed offsets beyond joint7).
        The original qpos is restored afterwards so the live simulation state
        is never corrupted.

        Args:
            q: (n_dof,) joint angles

        Returns:
            Tuple of:
                - position: (3,) [x, y, z] in world frame
                - quaternion: (4,) [w, x, y, z] orientation
        """
        # Save live simulation state before overwriting
        saved_qpos = self.data.qpos.copy()

        try:
            # Write candidate joint config (robot DOFs only; ball + other
            # free-joint DOFs at indices n_dof: are left untouched)
            self.data.qpos[:len(q)] = q
            # mj_kinematics propagates the full kinematic tree (xpos, xmat,
            # xipos, …) without computing forces, contacts, or actuators.
            # This is faster than mj_forward and safe to call mid-episode.
            m.mj_kinematics(self.model, self.data)

            # Prefer the configured site pose when available (e.g. paddle_contact)
            # so IK tracks the actual hitting surface, not just a body origin.
            if self.ee_site_id is not None:
                position = self.data.site_xpos[self.ee_site_id].copy()
                quaternion = np.zeros(4, dtype=np.float64)
                # Orientation can be sourced from a dedicated normal site so
                # IK, RL observation, and viewer debug arrows share one frame.
                quat_site_id = self.ee_normal_site_id if self.ee_normal_site_id is not None else self.ee_site_id
                m.mju_mat2Quat(quaternion, self.data.site_xmat[quat_site_id].copy())
            else:
                position = self.data.body(self.ee_body_name).xpos.copy()
                quaternion = self.data.body(self.ee_body_name).xquat.copy()
        finally:
            # Always restore — even if an exception is raised above
            self.data.qpos[:] = saved_qpos

        return position, quaternion


# ═══════════════════════════════════════════════════════════════════
# NUMERICAL IK SOLVER (OPTIMIZATION-BASED)
# ═══════════════════════════════════════════════════════════════════

class NumericalIKSolver(BaseIKSolver):
    """
    Numerical IK solver using nonlinear optimization.
    
    This solver formulates IK as an optimization problem:
        minimize: ||FK(q) - target_pos||² + w_ori * ||orientation_error||²
        subject to: q_min <= q <= q_max
    
    Where FK() is forward kinematics. We're trying to find joint angles q
    that minimize the distance between where the end-effector actually goes
    (FK(q)) and where we want it to go (target_pos).
    
    The solver uses SLSQP (Sequential Least Squares Programming), which is
    a gradient-based optimizer. It iteratively adjusts the joint angles to
    reduce the error, respecting the joint limits as constraints.
    
    Pros:
    ✅ Works for any robot (no analytical solution needed)
    ✅ Handles constraints naturally (joint limits, collision avoidance)
    ✅ Can prioritize position vs orientation via weights
    ✅ Generally robust and reliable
    
    Cons:
    ❌ Slower than analytical IK (typically 20-50ms per solve)
    ❌ May converge to local minima (not always the best solution)
    ❌ Requires good initial guess for best performance
    ❌ No guarantee of global optimum
    """
    
    def __init__(
        self,
        model: m.MjModel,
        data: m.MjData,
        end_effector_body: str,
        end_effector_site: t.Optional[str] = None,
        end_effector_normal_site: t.Optional[str] = None,
        position_weight: float = 1.0,
        orientation_weight: float = 0.1,
        joint_limit_margin: float = 0.01,
        max_iterations: int = 100,
    ):
        """
        Initialize numerical IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            end_effector_body: Name of end-effector body
            position_weight: Weight for position error (higher = prioritize position)
                           For table tennis, position is usually more important
                           than exact paddle orientation
            orientation_weight: Weight for orientation error
                              Set to 0 to ignore orientation completely
            joint_limit_margin: Safety margin from joint limits (radians)
                              Staying slightly away from limits prevents
                              singularities and gives some safety buffer
            max_iterations: Maximum optimization iterations
                          More iterations = potentially better solution but slower
        """
        super().__init__(
            model,
            data,
            end_effector_body,
            end_effector_site=end_effector_site,
            end_effector_normal_site=end_effector_normal_site,
        )
        
        self.w_pos = position_weight
        self.w_ori = orientation_weight
        self.joint_margin = joint_limit_margin
        self.max_iter = max_iterations
        
        # Get joint limits from model
        self.n_dof = 7  # FR3 robot has 7 DOF
        self.q_min = np.array([model.jnt_range[i, 0] for i in range(self.n_dof)])
        self.q_max = np.array([model.jnt_range[i, 1] for i in range(self.n_dof)])
    
    def solve(self, target_position: np.ndarray, target_orientation: t.Optional[np.ndarray] = None, initial_guess: t.Optional[np.ndarray] = None, target_normal: t.Optional[np.ndarray] = None) -> t.Tuple[np.ndarray, bool]:
        """
        Solve IK using numerical optimization.
        
        The optimization process works like this:
        1. Start from initial guess (current config if not provided)
        2. Compute forward kinematics to see where we are
        3. Calculate error (difference from target)
        4. Adjust joint angles to reduce error (gradient descent)
        5. Repeat until error is small enough or max iterations reached
        
        Args:
            target_position: (3,) [x, y, z] position
            target_orientation: (3,) [rx, ry, rz] Euler angles or (4,) quaternion (optional)
            target_normal: (3,) desired paddle face normal direction in world frame.
                           If provided, twist about the normal is ignored.
            initial_guess: (n_dof,) starting configuration (optional)
            
        Returns:
            Tuple of (joint_angles, success)
        """
        # Initial guess: use current config if not provided
        # A good initial guess is critical for fast convergence
        if initial_guess is None:
            q0 = self.data.qpos[:self.n_dof].copy()
        else:
            q0 = initial_guess.copy()
        
        # Ensure initial guess is within bounds (with safety margin)
        q0 = np.clip(q0, self.q_min + self.joint_margin,
                     self.q_max - self.joint_margin)
        
        # Convert target orientation to quaternion if provided.
        # If a normal is supplied, we only constrain the +Z axis alignment.
        target_quat = None
        target_normal_vec = None
        if target_normal is not None:
            target_normal = np.asarray(target_normal, dtype=float).reshape(-1)
            if target_normal.size != 3:
                raise ValueError("target_normal must be a 3-vector")
            n = float(np.linalg.norm(target_normal))
            if n < 1e-9:
                raise ValueError("target_normal must be non-zero")
            target_normal_vec = target_normal / n
        elif target_orientation is not None:
            target_orientation = np.asarray(target_orientation, dtype=float).reshape(-1)
            if target_orientation.size == 4:
                target_quat = target_orientation.copy()
                n = np.linalg.norm(target_quat)
                if n > 1e-9:
                    target_quat = target_quat / n
            elif target_orientation.size == 3:
                from ..utils.utils import euler_to_quaternion
                target_quat = euler_to_quaternion(
                    target_orientation[0],
                    target_orientation[1],
                    target_orientation[2],
                )
            else:
                raise ValueError("target_orientation must be Euler(3,) or quaternion(4,)")

        quat_mat = np.zeros(9, dtype=np.float64)
        
        # Define cost function for optimization
        def cost_function(q):
            """
            Optimization objective: minimize position and orientation error.
            
            This function is called many times during optimization. Each time,
            it evaluates how good a particular joint configuration is by
            computing forward kinematics and measuring the error.
            """
            # Forward kinematics: where does this config put the end-effector?
            current_pos, current_quat = self.forward_kinematics(q)
            
            # Position error (Euclidean distance)
            pos_error = np.linalg.norm(current_pos - target_position)**2
            
            # Orientation error (if target orientation specified)
            ori_error = 0.0
            if target_normal_vec is not None:
                # Align paddle face normal (body -Y axis) with target normal.
                # The visual mesh euler="1.5708 0 0" maps the blade face to body -Y.
                m.mju_quat2Mat(quat_mat, current_quat)
                current_z = -quat_mat.reshape(3, 3)[:, 1]   # body -Y
                dot = float(np.dot(current_z, target_normal_vec))
                dot = float(np.clip(dot, -1.0, 1.0))
                ori_error = np.arccos(dot) ** 2
            elif target_quat is not None:
                # Quaternion error using geodesic distance
                # This measures the angle between two orientations
                ori_error = self._quaternion_distance(current_quat, target_quat)**2
            
            # Joint limit penalty (soft constraint)
            # This gently pushes solutions away from joint limits
            limit_penalty = self._joint_limit_penalty(q)

            # Self-collision penalty: steer away from configs where paddle
            # or distal links approach proximal robot link bodies.
            self_collision_penalty = self._self_collision_penalty()
            
            # Total cost: weighted sum of all errors
            total_cost = (self.w_pos * pos_error +
                         self.w_ori * ori_error +
                         10.0 * limit_penalty +
                         self_collision_penalty)
            
            return total_cost
        
        # Joint limit bounds (hard constraint)
        # The optimizer will never try configurations outside these bounds
        bounds = [(self.q_min[i] + self.joint_margin,
                  self.q_max[i] - self.joint_margin)
                 for i in range(self.n_dof)]
        
        # Solve optimization problem using SLSQP
        result = minimize(
            cost_function,
            q0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': 1e-6}
        )
        
        # Check convergence using *actual* position error, not the combined
        # cost (which includes limit + self-collision penalties).
        # A 5 mm tolerance is sufficient for table tennis hitting tasks.
        final_pos, _ = self.forward_kinematics(result.x)
        pos_error_m  = float(np.linalg.norm(final_pos - target_position))
        success      = pos_error_m < 5e-3   # 5 mm

        return result.x, success
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute geodesic distance between two quaternions.
        
        This measures the angle of rotation needed to go from orientation q1
        to orientation q2. It's the proper way to measure orientation error
        because quaternions live on a 4D unit sphere, not in Euclidean space.
        
        Args:
            q1: (4,) quaternion [w, x, y, z]
            q2: (4,) quaternion [w, x, y, z]
            
        Returns:
            Angular distance in radians
        """
        # Normalize quaternions to ensure they're valid
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Dot product gives cosine of half the angle between orientations
        # We take absolute value because q and -q represent the same orientation
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, 0.0, 1.0)  # Numerical safety
        
        # Geodesic distance formula
        distance = 2 * np.arccos(dot)
        
        return distance
    
    def _joint_limit_penalty(self, q: np.ndarray) -> float:
        """
        Soft penalty for approaching joint limits.
        
        This encourages the optimizer to find solutions that stay comfortably
        within the joint limits, not right at the edge. Being near limits can
        cause problems like singularities and reduced maneuverability.
        
        The penalty increases quadratically as we approach the limits, gently
        pushing the solution toward the middle of the range.
        
        Args:
            q: (n_dof,) joint angles
            
        Returns:
            Penalty value (0 if far from limits, >0 near limits)
        """
        penalty = 0.0
        margin = 0.1  # Start penalizing at 10% of range from limit
        
        for i in range(self.n_dof):
            q_range = self.q_max[i] - self.q_min[i]
            margin_abs = margin * q_range
            
            # Lower limit penalty
            if q[i] < self.q_min[i] + margin_abs:
                dist = (self.q_min[i] + margin_abs - q[i]) / margin_abs
                penalty += dist**2
            
            # Upper limit penalty
            if q[i] > self.q_max[i] - margin_abs:
                dist = (q[i] - self.q_max[i] + margin_abs) / margin_abs
                penalty += dist**2
        
        return penalty

    def _self_collision_penalty(self) -> float:
        """
        Soft penalty for robot self-collision.

        After forward_kinematics() updates data.xpos, we check the world-frame
        positions of all body centroids.  Pairs that are kinematically adjacent
        (parent–child, grandparent–child, …) are excluded by the scene XML and
        are not of concern here.  We only need to prevent the distal bodies
        (paddle, fr3_link8, fr3_link7, fr3_link6) from approaching proximal
        bodies (fr3_link0 … fr3_link4) because those are the pairs that can
        actually collide when the arm folds toward itself.

        Approximate bounding-sphere radii are conservative (larger than the
        real geometry) so the penalty activates early and gives the optimizer
        room to steer away.

        Returns
        -------
        float
            Non-negative penalty value (0 when no bodies are close).
        """
        # (body_name, bounding_radius_m)
        PROXIMAL = [
            ("fr3_link0", 0.14),
            ("fr3_link1", 0.16),
            ("fr3_link2", 0.14),
            ("fr3_link3", 0.14),
            ("fr3_link4", 0.13),
        ]
        DISTAL = [
            ("paddle",     0.12),
            ("fr3_link8",  0.08),
            ("fr3_link7",  0.10),
            ("fr3_link6",  0.10),
        ]

        PENALTY_SCALE = 200.0   # strength multiplier

        penalty = 0.0
        try:
            for d_name, d_radius in DISTAL:
                d_pos = self.data.xpos[self.model.body(d_name).id]
                for p_name, p_radius in PROXIMAL:
                    p_pos = self.data.xpos[self.model.body(p_name).id]
                    dist = float(np.linalg.norm(d_pos - p_pos))
                    safe_dist = d_radius + p_radius          # combined radii
                    if dist < safe_dist:
                        violation = (safe_dist - dist) / safe_dist
                        penalty += PENALTY_SCALE * violation ** 2
        except Exception:
            pass   # body not found (shouldn't happen, but don't crash IK)

        return penalty


# ═══════════════════════════════════════════════════════════════════
# JACOBIAN-BASED IK SOLVER (ITERATIVE)
# ═══════════════════════════════════════════════════════════════════

class JacobianIKSolver(BaseIKSolver):
    """
    Jacobian-based iterative IK solver.
    
    This solver uses linearization of the forward kinematics around the
    current configuration. The Jacobian matrix J relates small changes in
    joint angles to small changes in end-effector position:
    
        Δx = J(q) * Δq
    
    Where:
    - Δx is a small change in end-effector position
    - Δq is a small change in joint angles
    - J(q) is the Jacobian matrix (depends on current configuration)
    
    The algorithm iteratively updates joint angles:
        q_new = q_old + α * J⁺ * (x_target - x_current)
    
    Where J⁺ is the pseudo-inverse of the Jacobian, and α is a step size.
    
    The key insight is that we're repeatedly solving a linear approximation
    of the nonlinear IK problem. Each iteration moves us closer to the target
    by taking a step in the direction that reduces error.
    
    Pros:
    ✅ Fast per-iteration (just matrix multiplication)
    ✅ Good for tracking tasks (small movements)
    ✅ Naturally handles redundant manipulators (7+ DOF)
    ✅ Simpler than full optimization
    
    Cons:
    ❌ May fail for large movements (linearization breaks down)
    ❌ Can get stuck in local minima
    ❌ Requires many iterations for convergence
    ❌ No explicit handling of joint limits or constraints
    """
    
    def __init__(
        self,
        model: m.MjModel,
        data: m.MjData,
        end_effector_body: str,
        end_effector_site: t.Optional[str] = None,
        end_effector_normal_site: t.Optional[str] = None,
        step_size: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ):
        """
        Initialize Jacobian-based IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            end_effector_body: Name of end-effector body
            step_size: Step size for each iteration (0 < α <= 1)
                      Smaller = more stable but slower convergence
                      Larger = faster but may overshoot or diverge
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (position error in meters)
                     When error drops below this, we consider it solved
        """
        super().__init__(
            model,
            data,
            end_effector_body,
            end_effector_site=end_effector_site,
            end_effector_normal_site=end_effector_normal_site,
        )
        
        self.alpha = step_size
        self.max_iter = max_iterations
        self.tol = tolerance
        self.n_dof = 7
        
        # Joint limits
        self.q_min = np.array([model.jnt_range[i, 0] for i in range(self.n_dof)])
        self.q_max = np.array([model.jnt_range[i, 1] for i in range(self.n_dof)])
    
    def solve(self, target_position: np.ndarray, target_orientation: t.Optional[np.ndarray] = None, initial_guess: t.Optional[np.ndarray] = None, target_normal: t.Optional[np.ndarray] = None) -> t.Tuple[np.ndarray, bool]:
        """
        Solve IK using Jacobian pseudo-inverse method.
        
        The algorithm:
        1. Start from initial configuration
        2. Compute current end-effector position via forward kinematics
        3. Calculate error vector (target - current)
        4. Compute Jacobian at current configuration
        5. Calculate joint angle update: Δq = α * J⁺ * error
        6. Update configuration: q = q + Δq
        7. Repeat until error is small or max iterations reached
        
        Note: This implementation only handles position, not orientation.
        Extending it to handle orientation would require using the angular
        velocity Jacobian as well.
        
        Args:
            target_position: (3,) [x, y, z] position
            target_orientation: Not used in this implementation
            target_normal: Not used in this implementation
            initial_guess: (n_dof,) starting configuration (optional)
            
        Returns:
            Tuple of (joint_angles, success)
        """
        # Initial guess
        if initial_guess is None:
            q = self.data.qpos[:self.n_dof].copy()
        else:
            q = initial_guess.copy()
        
        # Iterative optimization
        for iteration in range(self.max_iter):
            # Forward kinematics: where are we now?
            # (forward_kinematics saves/restores data.qpos internally)
            current_pos, _ = self.forward_kinematics(q)

            # Position error: how far from target?
            error = target_position - current_pos
            error_norm = np.linalg.norm(error)

            # Check convergence
            if error_norm < self.tol:
                return q, True  # Success!
            
            # Compute Jacobian at current configuration
            # This tells us how joint angles map to end-effector velocity
            J = self._compute_jacobian(q)
            
            # Pseudo-inverse: inverts the Jacobian while handling redundancy
            # For a 7-DOF arm reaching a 3D position, we have 4 degrees of
            # redundancy (infinite solutions). Pseudo-inverse picks the one
            # with minimum joint velocity.
            J_pinv = np.linalg.pinv(J)
            
            # Compute joint angle update
            # This is the direction in joint space that reduces position error
            dq = self.alpha * J_pinv @ error
            
            # Apply update with joint limit clamping
            q_new = q + dq
            q = np.clip(q_new, self.q_min, self.q_max)
        
        # Failed to converge within max iterations
        # Return best solution found, but mark as unsuccessful
        return q, False
    
    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix using MuJoCo.
        
        The Jacobian relates joint velocities to end-effector velocity:
            v_ee = J(q) * q̇
        
        MuJoCo provides an efficient function to compute this. The Jacobian
        is configuration-dependent, so we need to recompute it at each
        iteration as the joint angles change.
        
        Args:
            q: (n_dof,) joint configuration
            
        Returns:
            (3, n_dof) Jacobian matrix (position only, no orientation)
        """
        # Set configuration
        self.data.qpos[:self.n_dof] = q
        m.mj_forward(self.model, self.data)
        
        # Allocate Jacobian matrices
        # MuJoCo computes both position and rotation Jacobians
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian (what we want)
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian (not used here)
        
        # Get Jacobian from MuJoCo at site (preferred) or body origin.
        if self.ee_site_id is not None:
            m.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        else:
            m.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        
        # Extract only the columns corresponding to our robot joints
        # (In case there are other bodies/joints in the scene)
        J = jacp[:, :self.n_dof]
        return J
