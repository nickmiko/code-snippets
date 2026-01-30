#!/bin/bash

# Exit on error
set -e

# OS Detection
OS_TYPE=""
case "$(uname -s)" in
    Darwin)
        OS_TYPE='macos'
        ;;
    Linux)
        if [ -f /etc/os-release ]; then
            # shellcheck source=/dev/null
            . /etc/os-release
            if [ "$ID" = "ubuntu" ] || [ "$ID" = "pop" ] || [ "$ID_LIKE" = "debian" ]; then
                OS_TYPE='debian'
            fi
        fi
        ;;
esac

if [ -z "$OS_TYPE" ]; then
    echo "Unsupported OS. This script supports macOS and Debian-based Linux distributions."
    exit 1
fi

DEFAULT_PYTHON_VERSION="3.11.6"
PYENV_ENV_NAME="data-lab"
JUPYTER_KERNEL_NAME="python-data"
JUPYTER_KERNEL_DISPLAY_NAME="Python Data (pyenv)"

# --- Package Lists ---

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
  duckdb
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
)

install_apt_packages() {
  echo "Updating APT repository..."
  sudo apt-get update

  echo "Installing APT packages..."
  # Use a subshell to avoid polluting the environment with DEBIAN_FRONTEND
  (
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get install -y "${apt_packages[@]}"
  )

  # On Debian, 'fd' is installed as 'fdfind'. Create a symlink.
  if command -v fdfind &>/dev/null && ! command -v fd &>/dev/null; then
    echo "Creating symlink for fd -> fdfind..."
    sudo ln -s "$(command -v fdfind)" /usr/local/bin/fd
  fi
}

install_brew_formulas() {
  echo "Installing Homebrew formulas..."
  for formula in "${brew_formulas[@]}"; do
    if ! brew list "$formula" >/dev/null 2>&1; then
      echo "Installing $formula..."
      brew install "$formula"
    else
      echo "$formula already installed."
    fi
  done
}

install_brew_casks() {
  echo "Installing Homebrew casks..."
  for cask in "${brew_casks[@]}"; do
    if brew list --cask "$cask" >/dev/null 2>&1; then
      echo "$cask already installed."
      continue
    fi

    if existing_app_path=$(cask_existing_app "$cask"); then
      echo "$cask application already present at $existing_app_path. Skipping Homebrew install."
      continue
    fi

    echo "Installing $cask..."
    brew install --cask "$cask"
  done
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
    output = subprocess.check_output([
        "brew",
        "info",
        "--cask",
        "--json=v2",
        cask,
    ], stderr=subprocess.DEVNULL)
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

  if [ -z "$app_targets" ]; then
    return 1
  fi

  while IFS= read -r app_bundle; do
    [ -z "$app_bundle" ] && continue
    local full_path
    for full_path in "/Applications/$app_bundle" "$HOME/Applications/$app_bundle"; do
      if [ -d "$full_path" ]; then
        printf '%s\n' "$full_path"
        return 0
      fi
    done
  done <<< "$app_targets"

  return 1
}

ensure_zsh_plugins() {
  echo "Ensuring required Oh My Zsh plugins are configured..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not available; skipping plugin updates."
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

ensure_local_bin_path() {
  local target_dir="$HOME/.local/bin"
  local profile="$HOME/.zprofile"
  local export_line='export PATH="$HOME/.local/bin:$PATH"'

  mkdir -p "$target_dir"

  if [[ ":$PATH:" != *":$target_dir:"* ]]; then
    export PATH="$target_dir:$PATH"
    echo "Temporarily added $target_dir to PATH for this session."
  fi

  if [ ! -f "$profile" ]; then
    echo "Creating $profile to add PATH update..."
    {
      echo "# Created by setup.sh on $(date +%Y-%m-%d)"
      echo "$export_line"
    } > "$profile"
    return
  fi

  if grep -Fq "$target_dir" "$profile"; then
    echo "$target_dir already referenced in $profile."
    return
  fi

  echo "Adding $target_dir to PATH in $profile..."
  {
    echo ""
    echo "# Added by setup.sh on $(date +%Y-%m-%d)"
    echo "$export_line"
  } >> "$profile"
}

bootstrap_python_environment() {
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "pyenv not available; skipping Python environment bootstrap."
    return
  fi

  echo "Bootstrapping Python data environment via pyenv..."
  export PYENV_ROOT="$HOME/.pyenv"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"

  if [ ! -d "$PYENV_ROOT/versions/$DEFAULT_PYTHON_VERSION" ]; then
    echo "Installing Python $DEFAULT_PYTHON_VERSION..."
    pyenv install "$DEFAULT_PYTHON_VERSION"
  else
    echo "Python $DEFAULT_PYTHON_VERSION already installed."
  fi

  if [ ! -d "$PYENV_ROOT/versions/$PYENV_ENV_NAME" ]; then
    echo "Creating pyenv virtualenv $PYENV_ENV_NAME..."
    pyenv virtualenv "$DEFAULT_PYTHON_VERSION" "$PYENV_ENV_NAME"
  else
    echo "pyenv virtualenv $PYENV_ENV_NAME already exists."
  fi

  pyenv shell "$PYENV_ENV_NAME"
  pyenv global "$PYENV_ENV_NAME"
  pyenv rehash

  echo "Upgrading pip tooling..."
  python -m pip install --upgrade pip setuptools wheel

  echo "Installing core data science packages..."
  python -m pip install --upgrade \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    ipykernel

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
        sys.executable,
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        kernel_name,
        "--display-name",
        display_name,
    ])
    print(f"Installed Jupyter kernel '{display_name}'.")
PY
}

install_pipx_tools() {
  if ! command -v pipx >/dev/null 2>&1; then
    echo "pipx not available; skipping CLI tool installs."
    return
  fi

  echo "Installing Python CLI tools via pipx..."
  pipx ensurepath

  local tools=(black ruff mypy)
  for tool in "${tools[@]}"; do
    if pipx list --short | grep -qx "$tool"; then
      echo "$tool already installed with pipx."
    else
      echo "Installing $tool via pipx..."
      pipx install "$tool"
    fi
  done
}

echo "Starting setup for $OS_TYPE..."

if [ "$OS_TYPE" = "macos" ]; then
  # Install Homebrew if not installed
  if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    echo "Homebrew already installed."
  fi
elif [ "$OS_TYPE" = "debian" ]; then
  # On Debian, ensure curl and git are installed first as they are needed for next steps
  if ! command -v curl &>/dev/null || ! command -v git &>/dev/null; then
    echo "Installing curl and git..."
    sudo apt-get update
    sudo apt-get install -y curl git
  fi
fi

# Install Oh My Zsh if not installed
if [ ! -d "$HOME/.oh-my-zsh" ]; then
  echo "Installing Oh My Zsh..."
  # Use --unattended to prevent it from trying to change the shell and run zsh
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
else
  echo "Oh My Zsh already installed."
fi

# Install Zsh plugins
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"

# zsh-autosuggestions
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then
  echo "Installing zsh-autosuggestions..."
  git clone https://github.com/zsh-users/zsh-autosuggestions "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
fi

# zsh-syntax-highlighting
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then
  echo "Installing zsh-syntax-highlighting..."
  git clone https://github.com/zsh-users/zsh-syntax-highlighting.git "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
fi

# Install packages based on OS
if [ "$OS_TYPE" = "macos" ]; then
  install_brew_formulas
  install_brew_casks
elif [ "$OS_TYPE" = "debian" ]; then
  install_apt_packages
fi

# Install pyenv if not installed
if ! command -v pyenv &>/dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
else
    echo "pyenv already installed."
fi

bootstrap_python_environment
ensure_local_bin_path
install_pipx_tools
ensure_zsh_plugins

echo "Setup complete! Restart your terminal or run 'source ~/.zshrc' to apply changes."
