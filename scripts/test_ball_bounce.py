#!/usr/bin/env python3
"""
Ball Bounce Correctness Test
============================
Spawns the ball at each entry in `tested_initial_states` (config/simulation.yaml)
and runs the actual MuJoCo physics engine to verify the 2-bounce serve trajectory:

  Check 1 — Bounce 1 on opponent's table side  (x < 0, must hit table_top geom)
  Check 2 — Ball clears net  (crosses x = 0 at z > NET_TOP_Z, without touching net)
  Check 3 — Bounce 2 on robot's table side  (x > 0, must hit table_top geom)

Per-bounce physics telemetry is printed for each bounce:
  vz_before  — z-velocity just before contact  (negative = downward)
  vz_after   — z-velocity just after contact   (positive = upward)
  COR_direct — coefficient of restitution = |vz_after| / |vz_before|
  apex_z     — peak height of ball after the bounce
  rise       — apex_z minus ball-centre height at contact
  COR_apex   — COR inferred from apex height = sqrt(2·g·rise) / |vz_after|

Visual markers in the viewer:
  Orange sphere  — Bounce 1 position (opponent side)
  Blue sphere    — Net crossing height (at x = 0, actual y)
  Green sphere   — Bounce 2 position (robot side)
  Yellow sphere  — Apex of each bounce
  Red sphere     — Bounce on wrong side
  Magenta sphere — Ball touched net geom (illegal)

Usage:
  python3 scripts/test_ball_bounce.py                        # viewer on, 5 episodes
  python3 scripts/test_ball_bounce.py --episodes 3
  python3 scripts/test_ball_bounce.py --no-viewer
  python3 scripts/test_ball_bounce.py --slow 4               # 4× slower than real time
"""

import argparse
import os, sys, time
from contextlib import nullcontext
import numpy as np
import mujoco
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.gym_env import Environment
from src.utils.utils import load_config

# ── Config ───────────────────────────────────────────────────────────────────
_SIM_CFG   = load_config("config/simulation.yaml")
_ROBOT_CFG = load_config("config/robot.yaml")
_BALL_CFG  = _SIM_CFG["ball"]
_TABLE_CFG = _SIM_CFG["table"]

TABLE_Z      = _TABLE_CFG["position"][2] + 0.0125   # 0.7725 m  (table surface)
BALL_R       = _BALL_CFG["radius"]                  # 0.020 m
TABLE_HALF_X = _TABLE_CFG["length"] / 2.0           # 1.37 m
TABLE_HALF_Y = _TABLE_CFG["width"]  / 2.0           # 0.7625 m
NET_TOP_Z    = TABLE_Z + 0.1525                      # 0.9250 m
GROUND_Z     = _BALL_CFG["ground_z_threshold"]       # 0.01 m

# Height of ball centre when sitting on the table surface
CONTACT_Z = TABLE_Z + BALL_R                         # 0.7925 m

_G = 9.81  # m/s²

MAX_SIM_TIME = 5.0   # seconds

# Geom names from assets/scene.xml
BALL_GEOM  = "ball_geom"
TABLE_GEOM = "table_top"
NET_GEOMS  = {"net_collision", "net_top_edge"}

# Marker colours  (RGBA float32)
_ORANGE  = np.array([1.0, 0.55, 0.0,  0.95], dtype=np.float32)  # bounce 1 (opponent)
_BLUE    = np.array([0.1, 0.55, 1.0,  0.95], dtype=np.float32)  # net crossing
_GREEN   = np.array([0.1, 0.90, 0.1,  0.95], dtype=np.float32)  # bounce 2 (robot)
_RED     = np.array([1.0, 0.1,  0.1,  0.95], dtype=np.float32)  # wrong-side bounce
_MAGENTA = np.array([1.0, 0.1,  0.8,  0.95], dtype=np.float32)  # net touch
_YELLOW  = np.array([1.0, 0.95, 0.0,  0.95], dtype=np.float32)  # apex

_EYE9 = np.eye(3, dtype=np.float64).flatten()

# ── Load tested states ───────────────────────────────────────────────────────
TESTED_STATES = []
for _s in _BALL_CFG.get("tested_initial_states", []):
    try:
        TESTED_STATES.append({
            "position": np.asarray(_s["position"], dtype=float),
            "velocity": np.asarray(_s["velocity"], dtype=float),
            "spin":     np.asarray(_s.get("spin", [0.0, 0.0, 0.0]), dtype=float),
        })
    except Exception:
        pass


# ── Viewer helpers ────────────────────────────────────────────────────────────

def _draw_markers(viewer, markers):
    """
    Draw a list of sphere markers on the passive viewer's user_scn.
    Each marker: dict(pos=np.ndarray(3), rgba=np.ndarray(4), size=float).
    """
    if viewer is None:
        return
    try:
        scn = viewer.user_scn
        scn.ngeom = 0
        for m in markers:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, m["size"], dtype=np.float64),
                np.asarray(m["pos"],  dtype=np.float64),
                _EYE9,
                np.asarray(m["rgba"], dtype=np.float32),
            )
            scn.ngeom += 1
    except Exception:
        pass


def _sync(viewer, sleep_s):
    """Sync viewer; return False if viewer has been closed."""
    if viewer is None:
        return False
    try:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False
        viewer.sync()
        if sleep_s > 0:
            time.sleep(sleep_s)
        return True
    except Exception:
        return False


# ── Contact helpers ───────────────────────────────────────────────────────────

def _active_contacts(data, model):
    """Return set of geom names currently in contact with the ball."""
    hitting = set()
    for ci in range(data.ncon):
        c  = data.contact[ci]
        g1 = model.geom(int(c.geom1)).name
        g2 = model.geom(int(c.geom2)).name
        if BALL_GEOM in (g1, g2):
            other = g2 if g1 == BALL_GEOM else g1
            hitting.add(other)
    return hitting


# ── Apex telemetry helper ─────────────────────────────────────────────────────

def _finalize_apex(apex_idx, bounces, apex_per_bounce,
                   apex_z, apex_pos, viewer, vis_markers):
    """
    Compute and store apex telemetry for bounce `apex_idx` (1-indexed).
    Adds a yellow sphere at the apex position in the viewer.
    """
    if apex_pos is None or apex_z < TABLE_Z:
        return
    b    = bounces[apex_idx - 1]
    vz_b = b.get("vz_before") or 0.0
    vz_a = b.get("vz_after")  or 0.0
    rise = max(0.0, apex_z - CONTACT_Z)

    # COR from apex height: vz_after ≈ sqrt(2·g·rise)
    cor_apex   = np.sqrt(2.0 * _G * rise) / abs(vz_a) if vz_a else 0.0
    # COR direct: |vz_after| / |vz_before|
    cor_direct = abs(vz_a) / abs(vz_b) if vz_b else 0.0

    apex_per_bounce[apex_idx] = {
        "apex_z":    apex_z,
        "apex_pos":  apex_pos.copy(),
        "vz_before": vz_b,
        "vz_after":  vz_a,
        "rise":      rise,
        "cor_apex":  cor_apex,
        "cor_direct": cor_direct,
    }
    if viewer is not None:
        vis_markers.append({"pos": apex_pos.copy(), "rgba": _YELLOW, "size": 0.014})
        print(f"    [vis] APEX {apex_idx} marker at {_p3(apex_pos)}")


# ── Single-state test ─────────────────────────────────────────────────────────

def test_state(env, home_pos, state, viewer=None, slow_factor=1.0):
    """
    Run one ball-only episode.  When `viewer` is supplied the simulation is
    displayed at (real-time × slow_factor) speed with coloured sphere markers.

    Returns a dict with:
      bounces         — list of {pos, time, side, vz_before, vz_after}
      apex_per_bounce — dict of bounce-index → {apex_z, rise, cor_apex, cor_direct, ...}
      net_cross_z     — interpolated z at x=0 crossing (None if never crossed)
      net_hit         — True if ball physically touched a net geom
      b1_ok / nc_ok / b2_ok / valid — per-check pass flags
    """
    env.reset()
    env.set_robot_joints(home_pos, np.zeros(7))
    env.set_ball_state(state["position"].copy(),
                       state["velocity"].copy(),
                       spin=state["spin"].copy())
    mujoco.mj_forward(env.model, env.data)

    bounces        = []          # {pos, time, side, vz_before, vz_after}
    net_cross_z    = None
    net_cross_y    = None        # actual y where ball crosses x = 0
    net_hit        = False
    net_hit_pos    = None

    prev_pos       = state["position"].copy()
    prev_vel       = state["velocity"].copy()
    in_contact     = False       # table contact — previous step
    net_in_contact = False

    # Apex tracking state
    _tracking_apex = False       # True while tracking apex after a bounce
    _apex_idx      = None        # which bounce (1-indexed) we're tracking
    _apex_z_max    = -np.inf     # running max height since last bounce
    _apex_pos_max  = None
    apex_per_bounce: dict = {}   # bounce index → apex telemetry

    # Persistent visual markers — accumulate as events are detected
    vis_markers: list[dict] = []

    sleep_per_step = env.dt * slow_factor
    n_steps        = int(MAX_SIM_TIME / env.dt)

    for step in range(n_steps):
        ball = env.get_ball_state()
        pos  = ball["position"]
        vel  = ball["velocity"]
        t    = step * env.dt

        contacts = _active_contacts(env.data, env.model)

        # ── Table bounce (rising edge of table_top contact) ───────────────
        now_on_table = TABLE_GEOM in contacts
        if now_on_table and not in_contact:
            side = "opponent" if pos[0] < 0 else "robot" if pos[0] > 0 else "center"
            vz_b = float(prev_vel[2])     # z-vel just before contact (negative = downward)
            bounces.append({"pos": pos.copy(), "time": t, "side": side,
                            "vz_before": vz_b, "vz_after": None})

            # If we were tracking an apex, finalize it now (ball is bouncing again)
            if _tracking_apex and _apex_idx is not None and _apex_idx not in apex_per_bounce:
                _finalize_apex(_apex_idx, bounces, apex_per_bounce,
                               _apex_z_max, _apex_pos_max, viewer, vis_markers)
            _tracking_apex = False

            if viewer is not None:
                idx = len(bounces)
                if idx == 1:
                    color = _ORANGE if side == "opponent" else _RED
                    lbl   = f"B1 ({side})"
                elif idx == 2:
                    color = _GREEN if side == "robot" else _RED
                    lbl   = f"B2 ({side})"
                else:
                    color = _RED
                    lbl   = f"B{idx} ({side})"
                vis_markers.append({"pos": pos.copy(), "rgba": color, "size": 0.018})
                print(f"    [vis] {lbl} marker added at {_p3(pos)}")

        # ── Falling edge: ball just left the table ────────────────────────
        if in_contact and not now_on_table and bounces:
            # Capture vz just after leaving contact
            bounces[-1]["vz_after"] = float(vel[2])
            # Begin apex tracking for this bounce
            _tracking_apex = True
            _apex_idx      = len(bounces)
            _apex_z_max    = pos[2]
            _apex_pos_max  = pos.copy()

        in_contact = now_on_table

        # ── Apex tracking: update max height, finalize when ball descends ──
        if _tracking_apex and not now_on_table:
            if pos[2] > _apex_z_max:
                _apex_z_max   = pos[2]
                _apex_pos_max = pos.copy()
            elif (_apex_z_max > TABLE_Z
                  and pos[2] < _apex_z_max - 0.002
                  and _apex_idx not in apex_per_bounce):
                # Ball has descended ≥ 2 mm from its peak — apex is behind us
                _finalize_apex(_apex_idx, bounces, apex_per_bounce,
                               _apex_z_max, _apex_pos_max, viewer, vis_markers)
                _tracking_apex = False

        # ── Net contact ───────────────────────────────────────────────────
        now_on_net = bool(contacts & NET_GEOMS)
        if now_on_net and not net_in_contact:
            net_hit     = True
            net_hit_pos = pos.copy()
            if viewer is not None:
                vis_markers.append({"pos": pos.copy(), "rgba": _MAGENTA, "size": 0.018})
                print(f"    [vis] NET TOUCH marker added at {_p3(pos)}")
        net_in_contact = now_on_net

        # ── Net crossing: interpolate y and z at x = 0 ───────────────────
        if net_cross_z is None and prev_pos[0] < 0 < pos[0]:
            frac        = (0.0 - prev_pos[0]) / (pos[0] - prev_pos[0])
            net_cross_z = float(prev_pos[2] + frac * (pos[2] - prev_pos[2]))
            net_cross_y = float(prev_pos[1] + frac * (pos[1] - prev_pos[1]))
            if viewer is not None:
                cross_pos = np.array([0.0, net_cross_y, net_cross_z])
                vis_markers.append({"pos": cross_pos, "rgba": _BLUE, "size": 0.018})
                print(f"    [vis] NET CROSS marker at y={net_cross_y:.4f} z={net_cross_z:.4f} m")

        prev_pos = pos.copy()
        prev_vel = vel.copy()

        # ── Draw all accumulated markers every frame ───────────────────────
        _draw_markers(viewer, vis_markers)
        if not _sync(viewer, sleep_per_step):
            viewer = None   # viewer closed — continue headless

        # ── Termination ───────────────────────────────────────────────────
        if pos[2] < GROUND_Z:
            break
        if len(bounces) >= 2 and net_cross_z is not None:
            if step - int(bounces[1]["time"] / env.dt) > 200:
                break

        env._sim_step(home_pos)

    # Finalize apex tracking if the episode ended mid-flight
    if _tracking_apex and _apex_idx is not None and _apex_idx not in apex_per_bounce:
        _finalize_apex(_apex_idx, bounces, apex_per_bounce,
                       _apex_z_max, _apex_pos_max, viewer, vis_markers)

    # ── Hold markers for visual inspection ───────────────────────────────────
    if viewer is not None and vis_markers:
        print("    [vis] Holding markers for 3 s ...")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            _draw_markers(viewer, vis_markers)
            if not _sync(viewer, 0.016):   # ~60 fps hold
                break

    # ── Evaluate ──────────────────────────────────────────────────────────────
    b1 = bounces[0] if len(bounces) >= 1 else None
    b2 = bounces[1] if len(bounces) >= 2 else None

    b1_ok = b1 is not None and b1["side"] == "opponent"
    nc_ok = net_cross_z is not None and net_cross_z > NET_TOP_Z and not net_hit
    b2_ok = b2 is not None and b2["side"] == "robot"

    return {
        "bounces":         bounces,
        "apex_per_bounce": apex_per_bounce,
        "net_cross_z":     net_cross_z,
        "net_hit":         net_hit,
        "net_hit_pos":     net_hit_pos,
        "b1_ok":           b1_ok,
        "nc_ok":           nc_ok,
        "b2_ok":           b2_ok,
        "valid":           b1_ok and nc_ok and b2_ok,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pf(ok): return "PASS" if ok else "FAIL"
def _p3(v):  return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"
SEP  = "=" * 64
SEP2 = "-" * 64


def _print_bounce_physics(r):
    """Print per-bounce physics telemetry from a test_state() result."""
    for bidx, b in enumerate(r["bounces"], start=1):
        ap   = r["apex_per_bounce"].get(bidx)
        vz_b = b.get("vz_before")
        vz_a = b.get("vz_after")

        print(f"  Bounce {bidx} physics :")
        print(f"    position  : {_p3(b['pos'])} m   side={b['side']}")

        if vz_b is not None:
            print(f"    vz before : {vz_b:+.3f} m/s  (downward)")
        if vz_a is not None:
            cor_d = abs(vz_a) / abs(vz_b) if vz_b else 0.0
            print(f"    vz after  : {vz_a:+.3f} m/s  (COR_direct = {cor_d:.3f})")

        if ap is not None:
            print(f"    apex z    : {ap['apex_z']:.4f} m  "
                  f"(rise {ap['rise']:.4f} m above ball centre at contact)")
            print(f"    COR_apex  : {ap['cor_apex']:.3f}  "
                  f"(inferred from apex height)")
        else:
            print(f"    apex      : not measured")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ball bounce correctness test using MuJoCo physics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Run headless (no MuJoCo viewer window).",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, metavar="N",
        help="Number of episodes to run per tested state.",
    )
    parser.add_argument(
        "--slow", type=float, default=1.0, metavar="FACTOR",
        help="Slow-motion multiplier (e.g. 4 = 4× slower than real time).",
    )
    args = parser.parse_args()

    SHOW        = not args.no_viewer
    n_episodes  = args.episodes
    slow_factor = args.slow

    if not TESTED_STATES:
        print("ERROR: no tested_initial_states entries in config/simulation.yaml")
        sys.exit(1)

    home_pos = np.array(_ROBOT_CFG["robot"]["home_position"])
    env = Environment(scene_xml="assets/scene.xml", randomize=False)

    print(SEP)
    print("  BALL BOUNCE CORRECTNESS TEST  (MuJoCo physics)")
    print(SEP)
    print(f"  States         : {len(TESTED_STATES)}  (from config/simulation.yaml)")
    print(f"  Episodes/state : {n_episodes}")
    print(f"  Table surface  : z = {TABLE_Z:.4f} m")
    print(f"  Ball contact z : z = {CONTACT_Z:.4f} m  (table + ball radius)")
    print(f"  Net height     : z = {NET_TOP_Z:.4f} m  (ball must clear this at x = 0)")
    print(f"  Detection      : MuJoCo contact events on '{TABLE_GEOM}' geom")
    if SHOW:
        print(f"  Viewer         : ON  (slow_factor={slow_factor:.1f}×  |  use --no-viewer to skip)")
        print(f"  Marker key     : orange=bounce1  green=bounce2  blue=net-cross  "
              f"yellow=apex  red=wrong-side  magenta=net-touch")
    else:
        print(f"  Viewer         : OFF  (headless)")
    print(SEP)

    overall_pass = True

    try:
        viewer_cm = nullcontext(None)
        if SHOW:
            env.reset()
            viewer_cm = mujoco.viewer.launch_passive(env.model, env.data)

        with viewer_cm as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.90]
                viewer.cam.distance  = 7.5
                viewer.cam.elevation = -18
                viewer.cam.azimuth   = 55
                time.sleep(0.5)

            for i, state in enumerate(TESTED_STATES, start=1):
                p = state["position"]
                v = state["velocity"]
                w = state["spin"]

                print(f"\n  State {i}/{len(TESTED_STATES)}")
                print(f"    position : {_p3(p)} m")
                print(f"    velocity : {_p3(v)} m/s")
                print(f"    spin     : {_p3(w)} rad/s")

                ep_results = []
                for ep in range(1, n_episodes + 1):
                    print(f"  {SEP2}")
                    print(f"  Episode {ep}/{n_episodes}")
                    print(f"  {SEP2}")

                    r = test_state(env, home_pos, state,
                                   viewer=viewer, slow_factor=slow_factor)
                    ep_results.append(r)

                    b1 = r["bounces"][0] if len(r["bounces"]) >= 1 else None
                    b2 = r["bounces"][1] if len(r["bounces"]) >= 2 else None
                    nc = r["net_cross_z"]

                    # Check 1
                    if b1 is not None:
                        print(f"  Check 1 — Bounce 1 (opponent x<0) : "
                              f"{_p3(b1['pos'])}  t={b1['time']:.3f}s  "
                              f"side={b1['side']}  {_pf(r['b1_ok'])}")
                    else:
                        print(f"  Check 1 — Bounce 1 (opponent x<0) : NOT DETECTED  FAIL")

                    # Check 2
                    if nc is not None:
                        hit_note = "  [net physically touched — FAIL]" if r["net_hit"] else ""
                        print(f"  Check 2 — Net clearance            : "
                              f"z={nc:.4f} m (need > {NET_TOP_Z:.4f})  "
                              f"{_pf(r['nc_ok'])}{hit_note}")
                    else:
                        print(f"  Check 2 — Net clearance            : "
                              f"ball never crossed x = 0  FAIL")

                    # Check 3
                    if b2 is not None:
                        print(f"  Check 3 — Bounce 2 (robot   x>0)  : "
                              f"{_p3(b2['pos'])}  t={b2['time']:.3f}s  "
                              f"side={b2['side']}  {_pf(r['b2_ok'])}")
                    else:
                        print(f"  Check 3 — Bounce 2 (robot   x>0)  : NOT DETECTED  FAIL")

                    # Extra bounces
                    for j, bx in enumerate(r["bounces"][2:], start=3):
                        print(f"  Extra  — Bounce {j}               : "
                              f"{_p3(bx['pos'])}  t={bx['time']:.3f}s  side={bx['side']}")

                    # Per-bounce physics telemetry
                    if r["bounces"]:
                        print()
                        _print_bounce_physics(r)

                    ep_verdict = "VALID" if r["valid"] else "INVALID"
                    print(f"\n  Episode {ep} result: {ep_verdict}")

                    if SHOW and (ep < n_episodes or i < len(TESTED_STATES)):
                        time.sleep(0.5)

                # State-level aggregate (pass if every episode passed)
                n_pass = sum(r["valid"] for r in ep_results)
                state_ok = n_pass == n_episodes
                print(f"  {SEP2}")
                print(f"  State {i} summary : {n_pass}/{n_episodes} episodes VALID  "
                      f"{'PASS' if state_ok else 'FAIL'}")

                if not state_ok:
                    overall_pass = False

                if SHOW and i < len(TESTED_STATES):
                    print("  (1 s pause before next state ...)")
                    time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()

    print(f"\n{SEP}")
    print("  ALL STATES VALID" if overall_pass
          else "  SOME STATES INVALID — fix tested_initial_states in config/simulation.yaml")
    print(SEP)

    if SHOW:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
