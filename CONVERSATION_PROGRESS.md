# Table Tennis Project Progress

Date: 2026-04-07

## 1) Original Plan (Start of Conversation)

The conversation began with a phased validation plan:

1. Move robot/base constants into config and consume them from code.
2. Build a focused paddle-alignment test for a world-frame impact point and desired normal.
3. Add visual debugging (impact marker + target direction + paddle normal) in viewer.
4. Verify end-to-end action loop behavior:
   - Inverse kinematics
   - Trajectory generation
   - Joint-level control tracking
5. Port successful action-loop behavior into RL training path.
6. Update reward shaping to match intended game objective.

## 2) Features Added Throughout This Conversation

### A. Configuration and Environment Wiring

- Added robot base position in config and wired it into environment initialization.
- Environment now reads/validates home pose from config.

Files involved:
- config/robot.yaml
- src/rl/gym_env.py

### B. Paddle Alignment Test Script (Dedicated Debug Script)

- Created and iterated scripts/test_paddle_alignment.py to:
  - command a world-frame impact target and normal,
  - run IK + trajectory + control execution,
  - report scalar and per-axis position/normal errors,
  - support viewer/headless usage.
- Added visual overlays:
  - yellow impact marker,
  - green desired direction arrow,
  - blue paddle-normal arrow anchored to paddle contact.
- Added diagnostics for stage-by-stage debugging:
  - IK solution error,
  - trajectory target error,
  - joint tracking error at arrival.

File involved:
- scripts/test_paddle_alignment.py

### C. IK/Control Loop Refinements

- Extended IK usage to support normal-constrained solving in the action loop.
- Control pipeline now derives paddle normal from impact velocity direction when available:
  - desired normal = -normalize(impact velocity)
  - falls back to action normal when velocity is near zero.

File involved:
- src/rl/control_pipeline.py

### D. Comprehensive Script Visualization + Impact Direction Logic

- Added in-view debug vectors to scripts/test_comprehensive.py.
- Green vector now represents impact-direction tangent to predicted ball trajectory.
- Blue + red vectors show both paddle-face normals.
- White paddle rendering added for visual contrast.
- Predictor return shape was made consistent (4-value contract in all branches).

File involved:
- scripts/test_comprehensive.py

### E. RL Reward Model Changes (Per New Objective)

Implemented reward logic updates requested later in the conversation:

- Positive reward remains for valid paddle hit and far-side landing.
- Added positive bonus for hit using the opposite paddle face.
- Removed positive reward for net contact.
- Over-net transition is still tracked, but not positively rewarded.

Files involved:
- src/rl/reward.py
- src/rl/gym_env.py
- scripts/train_rl.py (evaluation metrics now include other-side hit rate)

## 3) Clarifications/Behavior Verified During Conversation

- scripts/test_comprehensive.py currently reports a fixed denominator (9/9) because summary is test-category based, not episode-count based.
- In scripts/test_comprehensive.py, tested states are used with deterministic cycling over valid entries, not random sampling among valid entries.

## 4) Where We Are Right Now

### Completed

- Config-driven base/home setup integrated.
- Dedicated alignment tool exists with rich diagnostics.
- Comprehensive visual debug overlays exist (impact direction + both paddle-face normals).
- RL action loop updated to use impact-velocity direction for target normal.
- RL reward model updated to:
  - reward post-bounce hits,
  - reward far-side landing,
  - add other-side-face hit bonus,
  - remove net-positive reward.

### Current Known Caveats

- Viewer teardown on this Linux/Wayland setup can still trigger GLFW/EGL instability in some runs (intermittent segfault behavior observed in interactive scripts).
- Controller-gain tuning attempts were made for alignment tracking, but this remains environment-dependent and should be re-validated on your machine with viewer enabled.

## 5) Suggested Next Validation Pass (Optional)

1. Run scripts/test_paddle_alignment.py for a matrix of normals and record IK/tracking errors.
2. Run scripts/test_comprehensive.py and confirm visual vectors align with expected impact dynamics.
3. Launch a short RL training run (scripts/train_rl.py) and inspect metrics:
   - hit_rate
   - other_side_hit_rate
   - land_rate
   - net_rate (diagnostic only)
4. If needed, tighten reward magnitudes after first learning curve review.

## 6) Diagrams

### A. Alignment Test Action Loop (scripts/test_paddle_alignment.py)

```mermaid
flowchart TD
  A[Input: impact xyz + normal xyz + t_arrive] --> B[Normalize desired normal]
  B --> C[IK solve with target position and target normal]
  C --> D[FK diagnostics: IK position/orientation error]
  D --> E[Generate min-jerk joint trajectory]
  E --> F[Per-step control command q_cmd]
  F --> G[Environment sim step]
  G --> H[Read paddle pose + paddle normal]
  H --> I[Viewer overlays: impact marker, green target dir, blue paddle normal]
  H --> J[Arrival diagnostics: position error, angular error, per-axis deltas]
```

### B. Comprehensive Script Data/Decision Flow (scripts/test_comprehensive.py)

```mermaid
flowchart TD
  A[Episode starts with sampled ball state] --> B[Physics predictor computes impact point/time/impact velocity]
  B --> C[Impact direction = normalize impact velocity]
  C --> D[Desired paddle normal = -impact direction]
  D --> E[IK target pose at contact offset]
  E --> F[Trajectory generation q_now -> q_goal]
  F --> G[Action loop sim stepping]
  G --> H[Ball event tracking: bounce, miss, over-net, landing]
  G --> I[Paddle metrics: closest approach, orientation/timing errors]
  G --> J[Viewer vectors: green impact direction, blue/red paddle normals]
```

### C. RL Training Loop (train_rl.py + gym_env.py + control_pipeline.py)

```mermaid
flowchart LR
  P[Policy action 10D] --> C1[ControlPipeline.plan]
  C1 --> C2[Decode impact position, time, impact velocity]
  C2 --> C3[Desired normal = -normalize impact velocity]
  C3 --> C4[IK solve + trajectory generation]
  C4 --> E1[Environment.step with action_repeat]
  E1 --> E2[Apply joint command each inner step]
  E2 --> E3[MuJoCo sim update]
  E3 --> R1[RewardCalculator.update]
  R1 --> R2[Sparse rewards: touch, other-side-hit bonus, far-side landing]
  R1 --> R3[No positive reward for net contact]
  E3 --> O1[Observation + info returned to policy]
  O1 --> P
```

### D. Reward Event Logic (Current)

```mermaid
flowchart TD
  A[Ball and paddle states] --> B{First bounce happened?}
  B -->|No| C[Early-hit path]
  B -->|Yes| D[Post-bounce contact path]

  D --> E{Ball-paddle contact?}
  E -->|Yes| F[+Touch reward]
  F --> G{Opposite paddle face?}
  G -->|Yes| H[+Other-side bonus]
  G -->|No| I[No side bonus]

  F --> J{Ball lands far side?}
  J -->|Yes| K[+Far-side reward + accuracy bonus]
  J -->|No| L[No landing bonus]

  M{Net contact?} --> N[Tracked for diagnostics only]
```
