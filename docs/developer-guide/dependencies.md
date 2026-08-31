# Managing dependencies

The dependencies are organized in a hierarchy:

- system dependencies. If any, these are satisfied through the system package manager.
- run dependencies, _e.g._ those specified in the `dependencies` array in
  `pyproject.toml` and which are required to run aurora in production. These
  can be satisfied through [uv].
- development dependencies, _e.g._ those specified in the
  `[dependency-groups]` table in `pyproject.toml`. These are **not**
  needed in production and can be satisfied through [uv].
- extra dependencies, _e.g._ those specified in the
  `[project.optional-dependencies]` table in `pyproject.toml` and that are
  required in production to enable optional features. These can also be satisfied
  with [uv].

## Why uv?

There are many command-line tools to interact with public and private Python package indexes.
[uv] is one of the newest around and it's a legitimate question to ask **why not
use simply [pip]?**

- [uv] has a lockfile format which is compact and multi-platform.
- [uv] allows granular control over _which index_ to use to gather _which
  dependency_ from. This is important when interacting with private indexes.
- [uv] is very fast.

## How do I handle dependencies?

What you would need to do is:

* Modify the `pyproject.toml` as needed.
* Run `uv sync` to re-install `src_method` with the modified dependency list and regenerate the lockfile.

[uv]: https://docs.astral.sh/uv/
