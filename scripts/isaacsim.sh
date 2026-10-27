#!/usr/bin/env bash
# Run the project CLI against an existing Isaac Sim binaries installation.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim-6.2.0}"
export CARB_APP_PATH="$ISAAC_PATH/kit"
export EXP_PATH="$ISAAC_PATH/apps"

if [[ ! -f "$ISAAC_PATH/setup_python_env.sh" ]]; then
    echo "Isaac Sim not found at $ISAAC_PATH. Set ISAAC_PATH to its installation directory." >&2
    exit 1
fi

source "$ISAAC_PATH/setup_python_env.sh"
exec "$PROJECT_ROOT/.venv/bin/isaaclab" "$@"
