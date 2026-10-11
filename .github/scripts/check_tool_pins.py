# ruff: noqa: INP001  (a standalone script, not part of a package)
"""Fail if a tool is pinned to different versions in pyproject.toml and prek.toml.

`uv run ruff` and the prek hook must agree, or a file can pass locally and fail
in CI. Each tool below is pinned as ``tool==X.Y.Z`` in a dependency group and as
``rev = "vX.Y.Z"`` on its pre-commit mirror.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIRRORS = {
    "ruff": "https://github.com/astral-sh/ruff-pre-commit",
    "ty": "https://github.com/astral-sh/ty-pre-commit",
}


def main() -> int:
    """Compare the pins and report every mismatch.

    Returns:
        The process exit code: 0 if every pin agrees, 1 otherwise.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    prek = tomllib.loads((ROOT / "prek.toml").read_text())
    requirements = [
        req
        for group in pyproject["dependency-groups"].values()
        for req in group
        if isinstance(req, str)
    ]
    revs = {repo["repo"]: repo.get("rev") for repo in prek["repos"]}

    errors = []
    for tool, mirror in MIRRORS.items():
        pins = {
            m.group(1)
            for req in requirements
            if (m := re.fullmatch(rf"{tool}==(\S+)", req.replace(" ", "")))
        }
        rev = revs.get(mirror)
        if len(pins) != 1 or rev is None:
            errors.append(f"{tool}: expected one '{tool}==' pin and a {mirror} rev")
        elif rev.removeprefix("v") != (pin := pins.pop()):
            errors.append(f"{tool}: pyproject.toml pins {pin}, prek.toml pins {rev}")

    for error in errors:
        print(error, file=sys.stderr)  # noqa: T201
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
