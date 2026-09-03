#!/usr/bin/env bash


# install project with all dependencies (no GPU extras in devcontainer)
uv sync --all-groups

# install the git hook shims
uv run prek install --prepare-hooks
