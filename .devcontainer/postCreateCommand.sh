#!/usr/bin/env bash

set -euxo pipefail


# install uv and uvx shell completions
echo 'eval "$(uv generate-shell-completion zsh)"' >> /home/vscode/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> /home/vscode/.zshrc
