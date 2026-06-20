#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  SCRIPT_SOURCED=1
else
  SCRIPT_SOURCED=0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PATH="${VIRTUAL_ENV:-/root/codescout-venv}"
REQUIRED_RG_VERSION="${REQUIRED_RG_VERSION:-15.1.0}"

info() {
  printf '[setup] %s\n' "$*"
}

fail() {
  printf '[setup:error] %s\n' "$*" >&2
  if [[ "$SCRIPT_SOURCED" == "1" ]]; then
    return 1
  fi
  exit 1
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$(pwd)" "$path"
  fi
}

VENV_PATH="$(abs_path "$VENV_PATH")"

case "$VENV_PATH" in
  "$REPO_ROOT"|"$REPO_ROOT"/*)
    fail "VIRTUAL_ENV must be outside the repository because Ray/SkyRL may copy the repo tree. Got: $VENV_PATH"
    ;;
esac

cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  info "uv not found; installing uv with the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  fail "uv is still not available. Add it to PATH, then rerun this script."
fi

if ! command -v cargo >/dev/null 2>&1; then
  info "cargo not found; installing Rust toolchain"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
elif [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

RG_NEEDS_INSTALL=1
if command -v rg >/dev/null 2>&1; then
  CURRENT_RG_VERSION="$(rg --version | head -n 1 | awk '{print $2}')"
  if [[ "$CURRENT_RG_VERSION" == "$REQUIRED_RG_VERSION" ]]; then
    RG_NEEDS_INSTALL=0
  else
    info "ripgrep version is $CURRENT_RG_VERSION; README uses $REQUIRED_RG_VERSION, reinstalling"
  fi
else
  info "ripgrep not found; installing version $REQUIRED_RG_VERSION"
fi

if [[ "$RG_NEEDS_INSTALL" == "1" ]]; then
  cargo install ripgrep --version "$REQUIRED_RG_VERSION" --locked
fi

info "creating/updating virtual environment at $VENV_PATH"
uv venv "$VENV_PATH"

export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VIRTUAL_ENV/bin:$PATH"

info "syncing CodeScout dependencies into the external virtual environment"
uv sync --all-extras --active

info "environment setup complete"
info "ripgrep: $(rg --version | head -n 1)"
info "python: $("$VIRTUAL_ENV/bin/python" --version)"
info "venv: $VIRTUAL_ENV"

if [[ -d "$REPO_ROOT/.venv" ]]; then
  info "warning: $REPO_ROOT/.venv still exists inside the repo. Remove it after confirming the external venv works:"
  info "  rm -rf $REPO_ROOT/.venv"
fi

if [[ "$SCRIPT_SOURCED" == "1" ]]; then
  # shellcheck disable=SC1091
  source "$VIRTUAL_ENV/bin/activate"
  info "activated $VIRTUAL_ENV in the current shell"
else
  info "activate it in your shell with:"
  info "  source $VIRTUAL_ENV/bin/activate"
fi
