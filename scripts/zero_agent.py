# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-action agent for this project's tasks (shows the robot's default pose)."""

# Warp captures ``enable_backward`` at module creation, so set it before importing Warp kernels.
import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import run_zero_agent_cli  # noqa: E402

import my_locomotion_training.tasks  # noqa: E402, F401


def main(argv: list[str] | None = None) -> int:
    return run_zero_agent_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
