# Successive Randomized Compression

[![Documentation](https://github.com/Algorithmiq/src-method/actions/workflows/docpages.yml/badge.svg)](https://docs.algorithmiq.fi/src-method)
[![Test src_method](https://github.com/Algorithmiq/src-method/actions/workflows/test.yml/badge.svg)](https://github.com/Algorithmiq/src-method/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An implementation of the SRC algorithm introduced in [Camaño, Epperly and Tropp, *Quantum* **10**, 2022 (2026)](https://doi.org/10.22331/q-2026-03-10-2022), extending the idea to other kinds of tensor networks.

### Features

The following primitives are supported:

1. MPO-MPS randomized contraction-compression.
2. MPO-MPO randomized contraction-compression.
3. MPO randomized compression.
4. MPS randomized compression.

`src_method` has no tensor-network framework dependency: it takes and returns plain lists of per-site NumPy arrays, one array per site.

```python
from src_method import apply, compress
```

The `apply` function covers cases 1 and 2 above, while the `compress` function covers cases 3 and 4. Both functions are pure, meaning no in-place modification ever happens. The user should
manage the assignment of the returned objects, possibly overwriting the input variables.
See the [reference documentation](algorithmiq.github.io/src_method/) for details, and the [tests](tests/) or [benchmarks](benches/) folders for usage examples.

Whether a train is an MPS or an MPO is inferred from the rank of its first site tensor, so no wrapper type is needed.

**NOTE**: the current implementation targets tensor networks with 3 or more sites. For smaller networks, an exact SVD-based fallback is dispatched, with a warning.

### Tensor Indexing Conventions

The array layout follows the default `quimb` tensor indexing conventions, so results round-trip through [Quimb](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/index.html) without any permutation:

```python
import quimb.tensor as qtn

result = qtn.MatrixProductOperator(apply(H1.arrays, H2.arrays, chi_out=64))
```

- **MPO Tensors:** Bulk tensors have index order `('l', 'r', 'u', 'd')`.
  Boundary tensors (at the edges) are rank-3, dropping the outer `'l'` or `'r'` index.

- **MPS Tensors:** Bulk tensors have index order `('l', 'r', 'u')`.
  Boundary tensors are rank-2, dropping the outer bond index.

Where `'l'`/`'r'` are left/right virtual bonds and `'u'`/`'d'` are the upper/lower physical legs.
Please keep this in mind when constructing or manipulating tensors directly.

### GPU Support

`src_method` runs on CPU (NumPy) by default and can be accelerated on GPUs via [CuPy](https://cupy.dev/). Both **NVIDIA** (CUDA) and **AMD** (ROCm) GPUs are supported through optional install extras.

At runtime, pass `device="gpu"` to use GPU acceleration. The library handles backend dispatch automatically.

## Installation

```bash
# CPU only (default)
uv pip install src_method

# With NVIDIA GPU support (CUDA 12.x)
uv pip install "src_method[gpu-nvidia]"

# With AMD GPU support (ROCm)
uv pip install "src_method[gpu-rocm]"
```

When including `src_method` in another project's `pyproject.toml`:

```toml
# CPU only
dependencies = ["src_method"]

# With NVIDIA GPU support
dependencies = ["src_method[gpu-nvidia]"]

# With AMD GPU support
dependencies = ["src_method[gpu-rocm]"]
```

## Setting up the development environment

The code has a [DevContainer] configuration that will get you up and running
with all dependencies installed and configured, including sane defaults for the
editor.

You will need:

1. A working [Docker] installation:
   - For macOS and Windows, install [Docker Desktop](https://docs.docker.com/get-docker/)
   - For Linux, install [Docker Engine](https://docs.docker.com/engine/install/#server) following the instructions for your specific distro.
2. The [Visual Studio Code] editor. A recent version is recommended, _e.g._ >=1.78
3. The VSCode [DevContainers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
4. The [GitHub CLI] tool.

You can clone the repository with:

```
gh repo clone Algorithmiq/src-method
```

We recommend using a Git credential manager, such as [GitHub CLI], configured to
use HTTPS as protocol for Git operations.

Once the code is locally available, you can open its containing folder in
[Visual Studio Code]. The editor will then set up the [DevContainer] for you.
The first time you open the folder the startup will take a few minutes. Once the
process is done, you will have _all_ project dependencies installed, including
the git hooks.
[Visual Studio Code] will be already configured with all the extensions helpful for Python development.

> [!TIP]
> The order in which Visual Studio Code loads the extensions in the
> DevContainer is non-deterministic.  You might have to execute the *Reload Window*
> command to get everything to work as expected after a fresh build of the container.

### Alternative: Nix flake

If you prefer [Nix] over Docker, the repository ships a `flake.nix` that provides
a development shell with [uv], [Git] and the [GitHub CLI], plus the native
libraries the binary wheels need at runtime. Python itself and all project
dependencies remain managed by `uv`.

With [flakes enabled](https://nixos.wiki/wiki/Flakes#Enable_flakes_temporarily), run:

```bash
nix develop
```

Entering the shell runs `uv sync --all-groups` and activates the uv project
environment (`$UV_PROJECT_ENVIRONMENT` if set, otherwise `.venv`), so you land
in a ready-to-use environment. GPU extras are not installed by the flake: add them
explicitly with `uv sync --all-groups --extra gpu-nvidia` (or `--all-groups --extra gpu-rocm`) on a machine with the matching drivers.

If you use [direnv], the provided `.envrc` enters the shell automatically:

```bash
direnv allow
```

Unlike the DevContainer, the Nix shell does not install the git hooks for
you. Run `prek install --prepare-hooks` once after the first `nix develop`.

## Documentation

We use [MkDocs] to generate our documentation pages. You can find the latest version [at this link].

We encourage you to build the documentation locally, so you can check that newer
documentation you might have added looks as it should.

To do so, open a terminal in [Visual Studio Code] and run:

```
mkdocs serve
```

the editor will prompt you to open a new page in your browser, where you can see the rendered documentation.

[DevContainer]: https://containers.dev/
[Docker]: https://docs.docker.com/get-docker/
[Visual Studio Code]: https://code.visualstudio.com/
[GitHub CLI]: https://cli.github.com/
[Git]: https://git-scm.com/
[Nix]: https://nixos.org/download/
[direnv]: https://direnv.net/
[uv]: https://docs.astral.sh/uv/
[MkDocs]: https://www.mkdocs.org/
[at this link]: https://docs.algorithmiq.fi/src_method
