"""
RewardCalculator
================

Stateful reward shaper for a single table tennis episode.

Call reset() at the start of each episode, then update() after every step.

Reward hierarchy (worst → best)
--------------------------------
    −20   No reaction      — robot barely moved + missed the ball
    −10   Not reached      — robot moved but ball passed without contact
     −3   Early hit        — paddle touched ball BEFORE first table bounce
                                                     (touch=+5, early_hit=−8)
        0   No hit           — ball bounced, robot didn't touch it
     +5   Touch            — paddle first contacts ball AFTER first bounce
     +8   Other-side touch — touch using opposite paddle face (extra +3)
    +18   Far table        — touch + ball lands on opponent's table (+accuracy)

Net contact and over-net transitions are tracked for diagnostics only.
They do not add positive reward.

All per-step dense terms are tiny regularisers that cannot reorder the levels:
  joint-limit penalty : max ≈ −0.001 / step
  jerk penalty        : max ≈ −0.0002 / step

Geometry reference
  Net:        x = 0.0
  Robot base: x = 1.5  (ball spawned at x ∈ [0.2, 0.8], travels toward +x)
  Far table:  x < 0
  Table surface z ≈ 0.7725 m
"""

import numpy as np
import typing as t


# ── Geometry constants ───────────────────────────────────────────────────────
NET_X        =  0.0
ROBOT_BASE_X =  1.5
TABLE_Z      =  0.7725
BALL_R       =  0.020

# ── Sparse reward values ─────────────────────────────────────────────────────
R_TOUCH         =  +5.0   # paddle contacts ball AFTER first bounce
R_TOUCH_OTHER_SIDE = +3.0 # extra reward when contact uses opposite face
R_NET_CONTACT   =   0.0   # tracked only; no positive reward
R_OVER_NET      =   0.0   # tracked only; no positive reward
R_FAR_SIDE      = +10.0   # additional: ball lands on far table      → cumulative +25
R_ACCURACY_MAX  =  +5.0   # landing accuracy bonus (on top of far_side, max +30 total)

R_MISS          = -10.0   # ball passes robot without contact (robot did move)
R_STATIONARY    =  -5.0   # extra penalty on top of miss when robot barely moved
                           # → "no reaction" total ≈ −10 + (−5) = −15 + tiny per-step ≈ −20
R_EARLY_HIT     =  -8.0   # hit before first bounce; net with R_TOUCH = +5 − 8 = −3

# ── Per-step regularisation (tiny — cannot reorder sparse levels) ────────────
R_LIMIT_MAX  = -0.001    # per-step, per-joint-violation
R_JERK_MAX   = -0.0002   # per-step

# ── Detection thresholds ──────────────────────────────────────────────────────
TOUCH_DIST_THRESHOLD      = 0.08   # m   — paddle within this → contact
STATIONARY_MOVE_THRESHOLD = 0.10   # rad — cumulative joint movement below this
                                   #       → episode treated as "no reaction"
JOINT_LIMIT_MARGIN        = 0.15   # rad — start penalising below this margin

Q_MIN = np.array([-2.9007, -1.8326, -2.9007, -3.0718, -2.8774,  0.4398, -3.0543])
Q_MAX = np.array([ 2.9007,  1.8326,  2.9007, -0.1169,  2.8774,  4.6251,  3.0543])

# Net geom names (from scene.xml) — used for ball↔net contact detection
_NET_GEOMS = {"net_collision", "net_top_edge"}
_BALL_GEOM = "ball_geom"


class RewardCalculator:
    """
    Stateful per-episode reward shaper.

    Parameters
    ----------
    env : Environment
        Live environment reference (used to read MuJoCo contact pairs).
    """

    def __init__(self, env):
        self.env = env
        self.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # Episode management
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset all per-episode state."""
        # Flags
        self._bounce_logged    = False
        self._hit_detected     = False
        self._hit_after_bounce = False   # True only when contact occurs after bounce
        self._hit_with_other_side = False
        self._net_contacted    = False   # ball contacted net geom after hit
        self._over_net         = False   # ball crossed x=0 going to far side
        self._landed_far_side  = False   # ball-table contact on far side (x<0)
        self._missed           = False   # ball passed robot without contact

        # Edge-trigger "just detected" flags (reset every step)
        self._hit_just_detected          = False
        self._other_side_hit_just_detected = False
        self._early_hit_just_detected    = False
        self._net_contact_just_detected  = False
        self._over_net_just_detected     = False
        self._landed_just_detected       = False
        self._miss_just_detected         = False

        # Trackers
        self._ball_prev_x: t.Optional[float] = None
        self._prev_joint_pos: t.Optional[np.ndarray] = None
        self._prev_joint_vel: t.Optional[np.ndarray] = None
        self._cumulative_joint_movement  = 0.0

        self._episode_reward = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Per-step update
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        ball_pos:    np.ndarray,
        ball_vel:    np.ndarray,
        paddle_pos:  np.ndarray,
        paddle_normal: np.ndarray,
        joint_pos:   np.ndarray,
        joint_vel:   np.ndarray,
        done_reason: t.Optional[str],
        step:        int,
    ) -> float:
        """
        Compute the shaped reward for one simulation step.

        Parameters
        ----------
        ball_pos    : (3,) ball world position
        ball_vel    : (3,) ball linear velocity
        paddle_pos  : (3,) paddle end-effector world position
        joint_pos   : (7,) joint angles
        joint_vel   : (7,) joint velocities
        done_reason : termination reason string, or None if episode continues
        step        : episode step counter (0-based)

        Returns
        -------
        float : reward for this step
        """
        r = 0.0

        # ── Track cumulative joint movement (stationary detection) ────────
        if self._prev_joint_pos is not None:
            self._cumulative_joint_movement += float(
                np.sum(np.abs(joint_pos - self._prev_joint_pos))
            )

        # ── Update event flags ────────────────────────────────────────────
        self._update_bounce_flag(ball_pos, ball_vel, step)
        self._update_hit_flag(ball_pos, paddle_pos, paddle_normal)
        self._update_net_contact_flag()
        self._update_over_net_flag(ball_pos)
        self._update_landed_flag(ball_pos, ball_vel)
        self._update_miss_flag(ball_pos)

        # ── Sparse: touch AFTER first bounce (+5) ────────────────────────
        if self._hit_just_detected and self._hit_after_bounce:
            r += R_TOUCH
            if self._other_side_hit_just_detected:
                r += R_TOUCH_OTHER_SIDE

        # ── Sparse: early hit penalty (−8); R_TOUCH (+5) already added ───
        # Net = +5 − 8 = −3  →  sits between "miss" (−10) and "no hit" (0)
        if self._early_hit_just_detected:
            r += R_EARLY_HIT

        # ── Sparse: ball contacts net after hit (+5; cumulative +10) ─────
        # Mutually exclusive with over_net.
        if self._net_contact_just_detected:
            r += R_NET_CONTACT

        # ── Sparse: ball crosses net to far side (+10; cumulative +15) ───
        if self._over_net_just_detected:
            r += R_OVER_NET

        # ── Sparse: ball lands on opponent's table (+10; cumulative +25) ─
        if self._landed_just_detected:
            r += R_FAR_SIDE
            # Accuracy bonus 0..+5 based on distance from ideal landing zone
            # centre (−0.5, 0.0); half-table length ≈ 1.37 m normalises dist.
            ideal_xy = np.array([-0.5, 0.0])
            acc_dist = float(np.linalg.norm(ball_pos[:2] - ideal_xy))
            r += R_ACCURACY_MAX * max(0.0, 1.0 - acc_dist / 1.37)

        # ── Sparse: miss (robot moved, −10) ──────────────────────────────
        if self._miss_just_detected:
            r += R_MISS

        # ── Sparse: terminal stationary penalty (−5 extra on top of miss) ─
        # Total "no reaction" = −10 (miss) + (−5) (stationary) + tiny per-step ≈ −20
        if (done_reason is not None
                and self._missed
                and not self._hit_detected
                and self._cumulative_joint_movement < STATIONARY_MOVE_THRESHOLD):
            r += R_STATIONARY

        # ── Dense: joint-limit regularisation (tiny, max ≈ −0.001/step) ──
        r += self._joint_limit_penalty(joint_pos)

        # ── Dense: jerk regularisation (tiny, max ≈ −0.0002/step) ────────
        if self._prev_joint_vel is not None:
            jerk = float(np.linalg.norm(joint_vel - self._prev_joint_vel))
            r += R_JERK_MAX * min(1.0, jerk / 50.0)

        # ── Save state ────────────────────────────────────────────────────
        self._prev_joint_pos = joint_pos.copy()
        self._prev_joint_vel = joint_vel.copy()
        self._ball_prev_x    = float(ball_pos[0])
        self._episode_reward += r
        return float(r)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal flag updaters
    # ─────────────────────────────────────────────────────────────────────────

    def _update_bounce_flag(self, ball_pos, ball_vel, step):
        """Detect first table bounce on robot's own side of the net."""
        if not self._bounce_logged and step > 5:
            if (ball_vel[2] > 0.3
                    and 0.74 < ball_pos[2] < 0.92
                    and ball_pos[0] > NET_X):
                self._bounce_logged = True

    def _update_hit_flag(self, ball_pos, paddle_pos, paddle_normal):
        """Detect first paddle–ball contact via proximity."""
        self._hit_just_detected       = False
        self._other_side_hit_just_detected = False
        self._early_hit_just_detected = False
        if self._hit_detected:
            return
        dist = float(np.linalg.norm(paddle_pos - ball_pos))
        if dist < TOUCH_DIST_THRESHOLD:
            self._hit_detected      = True
            self._hit_just_detected = True
            # Contact is on the opposite paddle face when the ball lies on
            # the negative side of the configured paddle normal.
            rel = ball_pos - paddle_pos
            if float(np.dot(rel, paddle_normal)) < 0.0:
                self._hit_with_other_side = True
                self._other_side_hit_just_detected = True
            if self._bounce_logged:
                self._hit_after_bounce = True
            else:
                # R_TOUCH (+5) fires via _hit_just_detected check in update(),
                # but _hit_after_bounce stays False, so the caller checks
                # _early_hit_just_detected and subtracts R_EARLY_HIT (−8).
                self._early_hit_just_detected = True

    def _update_net_contact_flag(self):
        """
        Detect ball contacting a net geom after the hit.

        Only fires once; once the ball is over the net this can no longer trigger.
        """
        self._net_contact_just_detected = False
        if self._net_contacted or not self._hit_detected or self._over_net:
            return
        try:
            for ci in range(self.env.data.ncon):
                c  = self.env.data.contact[ci]
                g1 = self.env.model.geom(int(c.geom1)).name
                g2 = self.env.model.geom(int(c.geom2)).name
                if _BALL_GEOM in (g1, g2) and _NET_GEOMS & {g1, g2}:
                    self._net_contacted             = True
                    self._net_contact_just_detected = True
                    return
        except Exception:
            pass

    def _update_over_net_flag(self, ball_pos):
        """Detect ball crossing x=0 toward the far side after being hit."""
        self._over_net_just_detected = False
        if (not self._over_net
                and self._hit_detected
                and not self._net_contacted
                and self._ball_prev_x is not None
                and self._ball_prev_x >= NET_X
                and ball_pos[0] < NET_X):
            self._over_net               = True
            self._over_net_just_detected = True

    def _update_landed_flag(self, ball_pos, ball_vel):
        """Detect ball landing on the opponent's table half."""
        self._landed_just_detected = False
        if (not self._landed_far_side
                and self._over_net
                and ball_pos[2] < TABLE_Z + BALL_R + 0.05
                and ball_vel[2] < -0.3
                and ball_pos[0] < NET_X):
            self._landed_far_side      = True
            self._landed_just_detected = True

    def _update_miss_flag(self, ball_pos):
        """Detect ball passing robot base x without contact."""
        self._miss_just_detected = False
        if (not self._hit_detected
                and not self._missed
                and self._ball_prev_x is not None
                and self._ball_prev_x <= ROBOT_BASE_X
                and ball_pos[0] > ROBOT_BASE_X):
            self._missed             = True
            self._miss_just_detected = True

    def _joint_limit_penalty(self, joint_pos: np.ndarray) -> float:
        """Tiny soft quadratic penalty for joints near limits (max ≈ −0.001/step)."""
        penalty = 0.0
        for i in range(len(joint_pos)):
            dist_lo = joint_pos[i] - Q_MIN[i]
            if dist_lo < JOINT_LIMIT_MARGIN:
                v = (JOINT_LIMIT_MARGIN - dist_lo) / JOINT_LIMIT_MARGIN
                penalty += R_LIMIT_MAX * v ** 2

            dist_hi = Q_MAX[i] - joint_pos[i]
            if dist_hi < JOINT_LIMIT_MARGIN:
                v = (JOINT_LIMIT_MARGIN - dist_hi) / JOINT_LIMIT_MARGIN
                penalty += R_LIMIT_MAX * v ** 2
        return penalty

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors (used by gym_env info dict)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    @property
    def hit_detected(self) -> bool:
        return self._hit_detected

    @property
    def hit_after_bounce(self) -> bool:
        return self._hit_after_bounce

    @property
    def hit_with_other_side(self) -> bool:
        return self._hit_with_other_side

    @property
    def net_contacted(self) -> bool:
        return self._net_contacted

    @property
    def over_net(self) -> bool:
        return self._over_net

    @property
    def landed_far_side(self) -> bool:
        return self._landed_far_side

    @property
    def missed(self) -> bool:
        return self._missed
