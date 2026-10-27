# my_locomotion_training

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `MyLocomotionTraining-Balance-Cartpole`
- `MyLocomotionTraining-Velocity-Flat-M20`
- `MyLocomotionTraining-Velocity-Rough-M20`


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## M20 wheel-legged locomotion

The M20 tasks follow Isaac Lab's Go2 velocity-task structure, but use mixed leg-position and wheel-velocity control.
They default to **Isaac Sim PhysX**, not Newton. With the existing Isaac Sim binaries in
`/home/yu/isaacsim-6.2.0`, no additional pip installation of Isaac Sim is necessary:

```bash
uv sync
# The wrapper defaults to $HOME/isaacsim-6.2.0; override ISAAC_PATH for another installation.
export ISAAC_PATH=/home/yu/isaacsim-6.2.0

# Short startup / PPO check, without a viewer
bash scripts/isaacsim.sh train --rl_library rsl_rl \
  --task MyLocomotionTraining-Velocity-Flat-M20 --num_envs 32 --max_iterations 2

# Start with flat terrain, then train a separate rough-terrain policy
bash scripts/isaacsim.sh train --rl_library rsl_rl \
  --task MyLocomotionTraining-Velocity-Flat-M20
bash scripts/isaacsim.sh train --rl_library rsl_rl \
  --task MyLocomotionTraining-Velocity-Rough-M20

# View the latest policy (checkpoint selection is scoped to the task's experiment directory)
bash scripts/isaacsim.sh play --rl_library rsl_rl \
  --task MyLocomotionTraining-Velocity-Flat-M20 --checkpoint latest --num_envs 16 --viz kit

# Real simulation regression checks (optional; needs GPU and network access on first use)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 M20_SIM_TESTS=1 \
  bash scripts/isaacsim.sh -p -m pytest -s tests/test_m20_simulation.py
```

The wrapper initializes the binary installation's Python/library paths and uses this project's `.venv`;
do not combine it with `uv run --extra isaacsim`, which selects a separate pip runtime. In this Isaac Lab
checkout, no viewer is launched unless requested with `--viz`. Both tasks default to **4096 environments**.
On an 8 GB GPU, this may exceed available VRAM; override with `--num_envs 128` if necessary.
Checkpoints and TensorBoard logs live under `logs/rsl_rl/m20_flat` and
`logs/rsl_rl/m20_rough`. Training defaults to 1500/3000 PPO iterations respectively.
The simulation test command disables unrelated auto-discovered pytest plugins (such as system ROS plugins).

**Known Newton limitation:** longer rough-terrain training runs with `physics=newton_mjwarp` have produced NaN
robot positions and velocities, despite passing short smoke runs. The exact solver/model interaction is still
unresolved; use the default Isaac Sim PhysX backend for M20 training. The subsequent observation NaN error is
a symptom of invalid physics state, not a reason to replace observations with zeros.
A 512-environment PhysX rough-terrain run completed 120 PPO iterations (1,474,560 transitions) without NaNs;
this is a stability check, not evidence of policy convergence.

### Inspected USD structure

Source: [Isaac Sim 6.2 M20 asset](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Robots/DeepRobotics/M20/M20.usd).
The small entry layer composes `configuration/M20_high_res_{base,physics,sensor,robot}.usd` through variants.
The task selects `Physics=PhysX`, `Sensor=None`, `Robot=Robot`; it does not load the authored camera/lidar sensors.
All asset layers remain remote and are cached by Isaac Lab; no third-party USD files are copied into this repository.
First startup therefore needs network access.

| Component | Authored structure / properties |
|-----------|--------------------------------|
| Root | `/M20`, meters, Z-up; floating articulation root on `base_link` |
| Links | 17 rigid bodies, total authored mass approximately 34.526 kg |
| Legs | `fl`, `fr`, `hl`, `hr`, each with `hipx_joint`, `hipy_joint`, `knee_joint` |
| Wheels | Four `*_wheel_joint`, continuous rotation, Y axis; body names `*_wheel` |
| Geometry | Two 0.25 m leg segments; cylindrical wheel collider radius 0.09 m, width 0.054 m |
| Drive limits | Legs 76.4 Nm, approximately 22.4 rad/s; wheels 21.6 Nm, approximately 79.3 rad/s |

USD angular limits/speeds are in degrees; the asset configuration converts speeds to rad/s. Joint-frame rotations
affect rotation signs: do not infer forward rolling from the `Y` axis token alone.
The initial front/rear hip and knee angles are mirrored, with the base at 0.49 m.
PD gains (legs 80/2, wheels 0/2) are baseline simulation choices, **not verified hardware controller parameters**.

### Actions, observations, and rewards

There are **16 actions**: 12 leg-position offsets (`0.25 rad * action`, added to the default pose, target clipped
to +/-1.5 rad), followed by four wheel speeds (`20 rad/s * action`, target clipped to +/-60 rad/s).
Within each group, joint order follows the articulation; the PhysX asset resolves legs by hipx, hipy, knee, each
in `fl, fr, hl, hr` order, and wheels in `fl, fr, hl, hr` order.

The flat policy observes **60 values**: base linear/angular velocity, projected gravity, velocity commands,
12 relative leg angles, 12 leg velocities, 16 previous actions, four wheel velocities, and four wheel-contact flags.
The rough policy adds **187 terrain-height samples** (247 values total). Continuous wheel angles are deliberately
excluded: spinning multiple revolutions must not change posture observations or position-limit penalties.
Actor and critic share this observation group.

Rewards retain velocity/yaw tracking, orientation, base height, effort, action smoothness, leg joint limits,
and non-wheel contact penalties. Wheel-specific terms penalize contact-point slip, wheel acceleration,
and wheel spin at zero translation **and** yaw commands. Slip uses wheel-body linear and angular velocity;
correct rolling is not penalized as sliding. Its vertical-radius approximation is most accurate for upright
wheels on level ground; steep slopes and heavily tilted wheels require further tuning.
Go2's feet-air-time reward is removed rather than encouraging wheels to leave the ground.
Rough terrain retains terrain-level curriculum; torso contact terminates an episode.

These are trainable starting configurations, not a supplied converged policy. Smoke tests establish that simulation,
observations, PPO updates, and checkpoint saving work; they do not establish locomotion quality or real-robot safety.

## Project structure

Task families live in `src/my_locomotion_training/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/cartpole`. Add another robot by
creating a sibling of `config/cartpole`; add another task by creating a sibling task-family directory.

## Project assets

Keep project-owned USD files and related data under `src/my_locomotion_training/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from my_locomotion_training.assets import MY_LOCOMOTION_TRAINING_ASSETS_DIR

robot_usd_path = MY_LOCOMOTION_TRAINING_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `my_locomotion_training.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
