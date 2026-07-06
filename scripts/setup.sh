#!/usr/bin/env bash

set -Eeuo pipefail

trap 'echo "❌ Error on line $LINENO: $BASH_COMMAND" >&2' ERR

# -----------------------------
# Defaults / configuration
# -----------------------------
DEFAULT_PYTHON_VERSION="3.11.6"
PYENV_ENV_NAME="data-lab"
JUPYTER_KERNEL_NAME="python-data"
JUPYTER_KERNEL_DISPLAY_NAME="Python Data (pyenv)"

SKIP_CASKS=0
SKIP_PYTHON=0
SKIP_ZSH=0
ONLY_PACKAGES=0
VERBOSE=0

OS_TYPE=""

brew_formulas=(
  git
  wget
  node
  python3
  htop
  tmux
  pyenv
  pyenv-virtualenv
  pipx
  jq
  yq
  ripgrep
  fd
  gnu-sed
  coreutils
  watch
  entr
  postgresql
  sqlite
  duckdb
  graphviz
  libomp
  gsl
  hdf5
  openssl@3
)

apt_packages=(
  git
  wget
  nodejs
  python3
  python3-pip
  python3-venv
  htop
  tmux
  jq
  yq
  ripgrep
  fd-find
  sed
  coreutils
  watch
  entr
  postgresql
  sqlite3
  graphviz
  libgsl-dev
  libhdf5-dev
  # pyenv dependencies
  build-essential
  libssl-dev
  zlib1g-dev
  libbz2-dev
  libreadline-dev
  libsqlite3-dev
  curl
  libncursesw5-dev
  xz-utils
  tk-dev
  libxml2-dev
  libxmlsec1-dev
  libffi-dev
  liblzma-dev
)

brew_casks=(
  iterm2
  rectangle
  visual-studio-code
  bitwarden
  proton-pass
  proton-mail
  steam
  cider
)

# -----------------------------
# Helpers
# -----------------------------
log() { echo "▶ $*"; }
warn() { echo "⚠ $*" >&2; }

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [options]

Options:
  --skip-casks      Skip GUI app installs (Homebrew casks on macOS, Flatpak apps on Linux).
  --skip-python     Skip pyenv/python environment bootstrap.
  --skip-zsh        Skip Oh My Zsh + plugin setup.
  --only-packages   Only install system/base packages and GUI apps, skip shells/python/pipx.
  --verbose         More verbose output.
  -h, --help        Show help.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-casks) SKIP_CASKS=1 ;;
      --skip-python) SKIP_PYTHON=1 ;;
      --skip-zsh) SKIP_ZSH=1 ;;
      --only-packages) ONLY_PACKAGES=1 ;;
      --verbose) VERBOSE=1 ;;
      -h|--help) usage; exit 0 ;;
      *)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done
}

retry() {
  local attempts="$1"; shift
  local delay="$1"; shift
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    warn "Command failed (attempt $n/$attempts). Retrying in $delay s..."
    sleep "$delay"
    n=$((n+1))
  done
}

append_unique_line() {
  local file="$1"
  local line="$2"
  touch "$file"
  if ! grep -Fqx "$line" "$file"; then
    echo "$line" >> "$file"
  fi
}

verify_checksum() {
  local file="$1"
  local expected_sha256="$2"
  local label="${3:-installer}"
  local actual_sha256
  if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256=$(sha256sum "$file" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    actual_sha256=$(shasum -a 256 "$file" | awk '{print $1}')
  else
    echo "❌ No SHA-256 utility found; cannot verify checksum for $label." >&2
    rm -f "$file"
    return 1
  fi
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "❌ SHA-256 mismatch for $label" >&2
    echo "   expected: $expected_sha256" >&2
    echo "   actual:   $actual_sha256" >&2
    echo "   The remote script may have changed. Review the new script before updating the expected checksum." >&2
    rm -f "$file"
    return 1
  fi
  [[ "$VERBOSE" -eq 1 ]] && log "✔ Checksum verified for $label"
}

detect_os() {
  case "$(uname -s)" in
    Darwin)
      OS_TYPE="macos"
      ;;
    Linux)
      if [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        case "${ID:-}" in
          ubuntu|pop|linuxmint|debian)
            OS_TYPE="debian"
            ;;
          *)
            case "${ID_LIKE:-}" in
              *debian*) OS_TYPE="debian" ;;
            esac
            ;;
        esac
      fi
      ;;
  esac

  if [[ -z "${OS_TYPE}" ]]; then
    echo "Unsupported OS. This script supports macOS and Debian-based Linux distributions." >&2
    exit 1
  fi
}

preflight_checks() {
  if [[ "${OS_TYPE}" == "debian" ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "sudo is required on Debian-based systems." >&2
      exit 1
    fi
    log "Validating sudo access..."
    sudo -v
  fi

  if [[ "${OS_TYPE}" == "macos" ]]; then
    if ! xcode-select -p >/dev/null 2>&1; then
      warn "Xcode Command Line Tools not found. Homebrew may prompt to install them."
    fi
  fi
}

ensure_local_bin_path() {
  local target_dir="$HOME/.local/bin"
  local profile="$HOME/.zprofile"
  local export_line='export PATH="$HOME/.local/bin:$PATH"'

  mkdir -p "$target_dir"

  if [[ ":$PATH:" != *":$target_dir:"* ]]; then
    export PATH="$target_dir:$PATH"
    log "Temporarily added $target_dir to PATH for this session."
  fi

  if [[ ! -f "$profile" ]]; then
    log "Creating $profile with PATH update..."
    {
      echo "# Created by setup.sh on $(date +%Y-%m-%d)"
      echo "$export_line"
    } > "$profile"
    return
  fi

  if grep -Fq "$target_dir" "$profile"; then
    log "$target_dir already referenced in $profile."
    return
  fi

  log "Adding $target_dir to PATH in $profile..."
  {
    echo ""
    echo "# Added by setup.sh on $(date +%Y-%m-%d)"
    echo "$export_line"
  } >> "$profile"
}

ensure_pyenv_shell_init() {
  local zshrc="$HOME/.zshrc"

  touch "$zshrc"

  append_unique_line "$zshrc" '# >>> pyenv init >>>'
  append_unique_line "$zshrc" 'export PYENV_ROOT="$HOME/.pyenv"'
  append_unique_line "$zshrc" 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
  append_unique_line "$zshrc" 'eval "$(pyenv init -)"'
  append_unique_line "$zshrc" 'eval "$(pyenv virtualenv-init -)"'
  append_unique_line "$zshrc" '# <<< pyenv init <<<'
}

install_apt_packages() {
  log "Updating APT repository..."
  retry 3 3 sudo apt-get update -y

  log "Installing APT packages..."
  (
    export DEBIAN_FRONTEND=noninteractive
    retry 3 3 sudo apt-get install -y --no-install-recommends "${apt_packages[@]}"
  )

  # On Debian, 'fd' is installed as 'fdfind'. Create/update a symlink.
  if command -v fdfind >/dev/null 2>&1; then
    if [[ ! -L /usr/local/bin/fd || "$(readlink /usr/local/bin/fd 2>/dev/null || true)" != "$(command -v fdfind)" ]]; then
      log "Ensuring symlink for fd -> fdfind..."
      sudo mkdir -p /usr/local/bin
      sudo ln -sf "$(command -v fdfind)" /usr/local/bin/fd
    fi
  fi
}

install_flatpak_if_needed() {
  if command -v flatpak >/dev/null 2>&1; then
    return 0
  fi

  log "Installing Flatpak..."
  if ! retry 3 3 sudo apt-get update -y; then
    warn "Failed to update apt before Flatpak install."
    return 1
  fi

  if ! retry 3 3 sudo apt-get install -y flatpak; then
    warn "Failed to install Flatpak."
    return 1
  fi

  # Optional integration package; safe to continue if unavailable.
  if ! sudo apt-get install -y gnome-software-plugin-flatpak; then
    warn "Could not install gnome-software-plugin-flatpak (optional)."
  fi

  return 0
}

ensure_flathub_remote() {
  if ! command -v flatpak >/dev/null 2>&1; then
    warn "flatpak command is unavailable; cannot configure Flathub."
    return 1
  fi

  if flatpak remote-list | awk '{print $1}' | grep -qx flathub; then
    return 0
  fi

  log "Adding Flathub remote..."
  if ! flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo; then
    warn "Failed to add Flathub remote."
    return 1
  fi

  return 0
}

install_flatpak_apps() {
  local failed=()

  if ! install_flatpak_if_needed; then
    warn "Skipping Flatpak app installs because Flatpak could not be set up."
    return 0
  fi

  if ! ensure_flathub_remote; then
    warn "Skipping Flatpak app installs because Flathub could not be configured."
    return 0
  fi

  local apps=(
    # Existing
    com.bitwarden.desktop
    me.proton.Pass
    me.proton.Mail
    com.valvesoftware.Steam
    sh.cider.Cider

    # Added recommendations
    io.dbeaver.DBeaverCommunity
    com.usebruno.Bruno
    me.hyliu.fluentreader
    io.github.jeffvli.Feishin
    md.obsidian.Obsidian
    org.libreoffice.LibreOffice
    org.gimp.GIMP
    org.inkscape.Inkscape
    com.github.tchx84.Flatseal
    com.mattjakeman.ExtensionManager
    com.github.wwmm.easyeffects
    org.videolan.VLC
    org.signal.Signal
    org.localsend.localsend_app
    com.syncthing.Syncthing
  )

  log "Installing Flatpak desktop apps..."
  local app
  for app in "${apps[@]}"; do
    if flatpak list --app --columns=application | grep -qx "$app"; then
      [[ "$VERBOSE" -eq 1 ]] && log "$app already installed."
      continue
    fi

    log "Installing $app..."
    if ! retry 2 3 flatpak install -y flathub "$app"; then
      warn "Failed to install Flatpak app: $app (continuing)"
      failed+=("$app")
      continue
    fi
  done

  if (( ${#failed[@]} > 0 )); then
    warn "Some Flatpak apps failed to install:"
    local f
    for f in "${failed[@]}"; do
      warn "  - $f"
    done
  else
    log "All Flatpak apps processed successfully."
  fi

  return 0
}

cask_existing_app() {
  local cask_name="$1"
  local app_targets

  if ! app_targets=$(CASK_NAME="$cask_name" python3 <<'PY'
import json
import os
import subprocess
import sys

cask = os.environ["CASK_NAME"]

try:
    output = subprocess.check_output(
        ["brew", "info", "--cask", "--json=v2", cask],
        stderr=subprocess.DEVNULL
    )
except subprocess.CalledProcessError:
    sys.exit(1)

try:
    data = json.loads(output)
except json.JSONDecodeError:
    sys.exit(1)

casks = data.get("casks", [])
if not casks:
    sys.exit(1)

targets = []
for artifact in casks[0].get("artifacts", []):
    if isinstance(artifact, str):
        if artifact.endswith(".app"):
            targets.append(artifact)
    elif isinstance(artifact, dict):
        app_value = artifact.get("app")
        if isinstance(app_value, str) and app_value.endswith(".app"):
            targets.append(app_value)
        elif isinstance(app_value, list):
            targets.extend([
                item for item in app_value
                if isinstance(item, str) and item.endswith(".app")
            ])

if not targets:
    sys.exit(1)

print("\n".join(targets))
PY
  ); then
    return 1
  fi

  [[ -z "$app_targets" ]] && return 1

  while IFS= read -r app_bundle; do
    [[ -z "$app_bundle" ]] && continue
    local full_path
    for full_path in "/Applications/$app_bundle" "$HOME/Applications/$app_bundle"; do
      if [[ -d "$full_path" ]]; then
        printf '%s\n' "$full_path"
        return 0
      fi
    done
  done <<< "$app_targets"

  return 1
}

install_brew_formulas() {
  log "Installing Homebrew formulas..."
  local formula
  for formula in "${brew_formulas[@]}"; do
    if brew list "$formula" >/dev/null 2>&1; then
      [[ "$VERBOSE" -eq 1 ]] && log "$formula already installed."
      continue
    fi
    log "Installing $formula..."
    retry 2 3 brew install "$formula"
  done
}

install_brew_casks() {
  log "Installing Homebrew casks..."
  local cask
  for cask in "${brew_casks[@]}"; do
    if brew list --cask "$cask" >/dev/null 2>&1; then
      [[ "$VERBOSE" -eq 1 ]] && log "$cask already installed."
      continue
    fi

    if existing_app_path=$(cask_existing_app "$cask"); then
      log "$cask app already present at $existing_app_path. Skipping Homebrew install."
      continue
    fi

    log "Installing $cask..."
    if ! retry 2 3 brew install --cask "$cask"; then
      warn "Failed to install cask $cask after retries."
      _cask_failed=1
    fi
  done
  return "${_cask_failed:-0}"
}

install_oh_my_zsh_and_plugins() {
  if [[ -d "$HOME/.oh-my-zsh" ]]; then
    log "Oh My Zsh already installed."
  else
    log "Installing Oh My Zsh..."
    # Pinned to master branch. If the checksum fails after an upstream update:
    #   1. Review the diff at https://github.com/ohmyzsh/ohmyzsh/commits/master/tools/install.sh
    #   2. Confirm the changes are legitimate before trusting the new script.
    #   3. Update the SHA-256 below: curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sha256sum
    local omz_installer omz_sha256="95118b50d062198597e2b73d3a57b609fd95ca68cdc86faf4460d955f0172b61"
    omz_installer=$(mktemp)
    retry 2 3 curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -o "$omz_installer"
    verify_checksum "$omz_installer" "$omz_sha256" "Oh My Zsh installer"
    chmod +x "$omz_installer"
    "$omz_installer" --unattended
    rm -f "$omz_installer"
  fi

  local zsh_custom="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
  mkdir -p "$zsh_custom/plugins"

  if [[ ! -d "$zsh_custom/plugins/zsh-autosuggestions" ]]; then
    log "Installing zsh-autosuggestions..."
    retry 2 3 git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions "$zsh_custom/plugins/zsh-autosuggestions"
  fi

  if [[ ! -d "$zsh_custom/plugins/zsh-syntax-highlighting" ]]; then
    log "Installing zsh-syntax-highlighting..."
    retry 2 3 git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git "$zsh_custom/plugins/zsh-syntax-highlighting"
  fi
}

ensure_zsh_plugins() {
  log "Ensuring required Oh My Zsh plugins are configured..."
  if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not available; skipping plugin updates."
    return
  fi

  python3 <<'PY'
from pathlib import Path
from datetime import datetime

zshrc = Path.home() / ".zshrc"
required = ["git", "zsh-autosuggestions", "zsh-syntax-highlighting"]

if not zshrc.exists():
    zshrc.write_text(
        f"# Created by setup.sh on {datetime.now():%Y-%m-%d}\n"
        f"plugins=({' '.join(required)})\n"
    )
    print("Created ~/.zshrc with required plugins.")
    raise SystemExit

text = zshrc.read_text()
plugins_line = None
for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("plugins=(") and stripped.endswith(")"):
        plugins_line = line
        break

if plugins_line is None:
    with zshrc.open("a") as fh:
        fh.write("\n# Added by setup.sh on {}\n".format(datetime.now().strftime("%Y-%m-%d")))
        fh.write("plugins=({})\n".format(" ".join(required)))
    print("Appended plugins line to ~/.zshrc.")
    raise SystemExit

existing = [p for p in plugins_line.replace("plugins=(", "").rstrip(")").split() if p]
missing = [p for p in required if p not in existing]

if not missing:
    print("Required plugins already present in ~/.zshrc.")
    raise SystemExit

updated_line = "plugins=(" + " ".join(existing + missing) + ")"
zshrc.write_text(text.replace(plugins_line, updated_line, 1))
print("Updated plugins line in ~/.zshrc.")
PY
}

install_pyenv_if_missing() {
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  export PATH="$PYENV_ROOT/bin:$PATH"

  # If pyenv is already runnable, we're done.
  if command -v pyenv >/dev/null 2>&1; then
    log "pyenv already installed."
    return
  fi

  # If ~/.pyenv exists, don't run installer again (it will fail).
  if [[ -d "$PYENV_ROOT" ]]; then
    warn "$PYENV_ROOT exists but pyenv command is not available yet."
    warn "Attempting to use existing installation..."

    if [[ -x "$PYENV_ROOT/bin/pyenv" ]]; then
      log "Found existing pyenv at $PYENV_ROOT."
      return
    fi

    cat >&2 <<EOF
pyenv directory exists but pyenv is still not runnable.
Try one of these:
  1) Open a new shell and run again
  2) source ~/.zshrc  (or ~/.bashrc), then run again
  3) If installation is corrupted: rm -rf "$PYENV_ROOT" and rerun setup
EOF
    return 1
  fi

  # Fresh install path
  log "Installing pyenv..."
  # Pinned to pyenv-installer master branch. If the checksum fails after an upstream update:
  #   1. Review the diff at https://github.com/pyenv/pyenv-installer/commits/master/bin/pyenv-installer
  #   2. Confirm the changes are legitimate before trusting the new script.
  #   3. Update the SHA-256 below: curl -fsSL https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | sha256sum
  local pyenv_installer pyenv_sha256="4b0adf623a6205727163eb98610b6c5e63f23b99183948b874d867cd9b30ef13"
  pyenv_installer=$(mktemp)
  retry 2 3 curl -fsSL https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer -o "$pyenv_installer"
  verify_checksum "$pyenv_installer" "$pyenv_sha256" "pyenv installer"
  bash "$pyenv_installer"
  rm -f "$pyenv_installer"

  # Ensure current shell can see pyenv immediately
  export PATH="$PYENV_ROOT/bin:$PATH"
  if ! command -v pyenv >/dev/null 2>&1; then
    warn "pyenv installation completed but command not found in current shell."
    warn "Open a new terminal (or source your shell rc file) and rerun."
    return 1
  fi
}

bootstrap_python_environment() {
  if [[ "$SKIP_PYTHON" -eq 1 ]]; then
    log "Skipping Python bootstrap (--skip-python)."
    return
  fi

  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"

  if ! command -v pyenv >/dev/null 2>&1; then
    warn "pyenv not available; skipping Python environment bootstrap."
    return
  fi

  log "Bootstrapping Python data environment via pyenv..."
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"

  if [[ ! -d "$PYENV_ROOT/versions/$DEFAULT_PYTHON_VERSION" ]]; then
    log "Installing Python $DEFAULT_PYTHON_VERSION..."
    retry 2 3 pyenv install "$DEFAULT_PYTHON_VERSION"
  else
    log "Python $DEFAULT_PYTHON_VERSION already installed."
  fi

  if [[ ! -d "$PYENV_ROOT/versions/$PYENV_ENV_NAME" ]]; then
    log "Creating pyenv virtualenv $PYENV_ENV_NAME..."
    pyenv virtualenv "$DEFAULT_PYTHON_VERSION" "$PYENV_ENV_NAME"
  else
    log "pyenv virtualenv $PYENV_ENV_NAME already exists."
  fi

  pyenv shell "$PYENV_ENV_NAME"
  pyenv rehash

  log "Upgrading pip tooling..."
  python -m pip install --upgrade pip setuptools wheel

  log "Installing core data science packages..."
  python -m pip install --upgrade \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    ipykernel \
    duckdb

  JUPYTER_KERNEL_NAME="$JUPYTER_KERNEL_NAME" \
  JUPYTER_KERNEL_DISPLAY_NAME="$JUPYTER_KERNEL_DISPLAY_NAME" \
  python <<'PY'
import json
import subprocess
import sys
import os

kernel_name = os.environ["JUPYTER_KERNEL_NAME"]
display_name = os.environ["JUPYTER_KERNEL_DISPLAY_NAME"]

try:
    output = subprocess.check_output(["jupyter", "kernelspec", "list", "--json"], stderr=subprocess.STDOUT)
    data = json.loads(output)
    kernels = data.get("kernelspecs", {})
except Exception:
    kernels = {}

if kernel_name in kernels:
    print(f"Jupyter kernel '{display_name}' already present.")
else:
    subprocess.check_call([
        sys.executable, "-m", "ipykernel", "install", "--user",
        "--name", kernel_name, "--display-name", display_name
    ])
    print(f"Installed Jupyter kernel '{display_name}'.")
PY
}

install_pipx_tools() {
  if [[ "$SKIP_PYTHON" -eq 1 ]]; then
    log "Skipping pipx tools because Python bootstrap is skipped."
    return
  fi

  if ! command -v pipx >/dev/null 2>&1; then
    warn "pipx not available; skipping CLI tool installs."
    return
  fi

  log "Installing Python CLI tools via pipx..."
  pipx ensurepath

  local tools=(black ruff mypy)
  local tool
  for tool in "${tools[@]}"; do
    if pipx list --short | grep -qx "$tool"; then
      [[ "$VERBOSE" -eq 1 ]] && log "$tool already installed with pipx."
    else
      log "Installing $tool via pipx..."
      pipx install "$tool"
    fi
  done
}

setup_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    log "Homebrew already installed."
  else
    log "Installing Homebrew..."
    # Pinned to Homebrew install HEAD. If the checksum fails after an upstream update:
    #   1. Review the diff at https://github.com/Homebrew/install/commits/HEAD/install.sh
    #   2. Confirm the changes are legitimate before trusting the new script.
    #   3. Update the SHA-256 below: curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | sha256sum
    local brew_installer brew_sha256="99287f194a8b3c9e6b0203a11a5fa54518be57209343e6bb954dec4635796d9d"
    brew_installer=$(mktemp)
    retry 2 3 curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh -o "$brew_installer"
    verify_checksum "$brew_installer" "$brew_sha256" "Homebrew installer"
    chmod +x "$brew_installer"
    /bin/bash "$brew_installer"
    rm -f "$brew_installer"
  fi

  # Load brew shellenv for current session and future shells.
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    append_unique_line "$HOME/.zprofile" 'eval "$(/opt/homebrew/bin/brew shellenv)"'
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
    append_unique_line "$HOME/.zprofile" 'eval "$(/usr/local/bin/brew shellenv)"'
  fi
}

setup_linux_base_tools() {
  if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    log "Installing curl and git..."
    retry 3 3 sudo apt-get update -y
    retry 3 3 sudo apt-get install -y curl git
  fi
}

# -----------------------------
# Main
# -----------------------------
main() {
  parse_args "$@"
  detect_os
  preflight_checks

  log "Starting setup for $OS_TYPE..."

  if [[ "$OS_TYPE" == "macos" ]]; then
    setup_homebrew
    install_brew_formulas
    if [[ "$SKIP_CASKS" -eq 0 ]]; then
      if ! install_brew_casks; then
        warn "Some casks failed to install; continuing."
      fi
    else
      log "Skipping casks (--skip-casks)."
    fi
  else
    setup_linux_base_tools
    install_apt_packages
    if [[ "$SKIP_CASKS" -eq 0 ]]; then
      install_flatpak_apps
    else
      log "Skipping GUI app installs (--skip-casks)."
    fi
  fi

  if [[ "$ONLY_PACKAGES" -eq 1 ]]; then
    log "Done (--only-packages)."
    exit 0
  fi

  if [[ "$SKIP_ZSH" -eq 0 ]]; then
    install_oh_my_zsh_and_plugins
    ensure_zsh_plugins
  else
    log "Skipping zsh setup (--skip-zsh)."
  fi

  install_pyenv_if_missing
  ensure_pyenv_shell_init
  bootstrap_python_environment

  ensure_local_bin_path
  install_pipx_tools

  log "✅ Setup complete! Restart your terminal or run 'source ~/.zshrc' to apply changes."
}

main "$@"