# SRC Method

## Successive Randomized Compression

An implementation of the SRC algorithm introduced on [arXiv:2504.06475](https://arxiv.org/abs/2504.06475), but extending the idea to other kinds of tensor networks.

### Features

The following primitives are supported:

1. MPO-MPS randomized contraction-compression.
2. MPO-MPO randomized contraction-compression.
3. MPO randomized compression.
4. MPS randomized compression.
5. Randomized contraction-compression of a whole stack of trains in one sweep:
   `MPO^k`, `MPO^k . MPS` and `MPS . MPO^k`.

`src_method` has no tensor-network framework dependency: it takes and returns plain lists of per-site NumPy arrays, one array per site.

```python
from src_method import apply, compress, src
```

The `apply` function covers cases 1 and 2 above, the `compress` function cases 3 and 4, and `src` all five: `apply` and `compress` are its two- and one-train special cases. All three functions are pure, meaning no in-place modification ever happens. The user should
manage the assignment of the returned objects, possibly overwriting the input variables.
See the [reference documentation](algorithmiq.github.io/src_method/) for details, and the [tests](../tests/) or [benchmarks](../benches/) folders for usage examples.

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

### Contraction Conventions

`src` contracts a *stack* of trains and compresses the result in a single sweep:

```python
from src_method import src

state = src(U3, U2, U1, psi, chi_out=64)  # U3 U2 U1 |psi>, U1 acts first
```

The stack is written in mathematical order. Each contraction joins the `'d'` leg of a train with the `'u'` leg (or the physical leg of an MPS) of the train to its right:

| Stack | Contraction | Result |
|---|---|---|
| `src(A)`, `src(psi)` | none | compressed MPO or MPS |
| `src(A, B, ...)` | `A.d` with `B.u` | MPO |
| `src(A, ..., psi)` | `A.d` with `psi` | MPS on the `'u'` leg of `A` (a ket) |
| `src(phi, A, ...)` | `phi` with `A.u` | MPS on the `'d'` leg of the last MPO (a bra) |

An MPS may appear only first or last, and not both. `apply(A, B)` and `compress(A)` are the two- and one-train cases; `apply` accepts only an MPO on the left.

A leading MPS is a row vector used **without conjugation**: `src(phi, A, B)` computes `phiᵀ A B`, which equals `src(Bᵀ, Aᵀ, phi)` with `ᵀ` swapping the `'u'` and `'d'` legs. For the physical bra `<psi| A B`, conjugate first:

```python
bra = src([t.conj() for t in psi], A, B, chi_out=64)
```

The result then pairs with a ket by plain contraction, with no further conjugation.

**Cost.** The per-site cost of the sweep grows with `chi_out**2` times the product of the bond dimensions of the layers. Compressing a whole stack at once pays off for shallow stacks of thin layers, such as two or three Trotter layers; apply anything else pairwise. See [the stack depth benchmarks](https://github.com/Algorithmiq/src-method/tree/main/benches/stack) for measurements.

## Installation

```bash
# CPU only (default)
uv pip install src_method

# With NVIDIA GPU support (CUDA 12.x)
uv pip install "src_method[gpu-nvidia]"

# With AMD GPU support (ROCm)
uv pip install "src_method[gpu-rocm]"
```

Or simply add it to the `dependencies` of your project's `pyproject.toml`.

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
4. (Optional, but **highly recommended**) The [GitHub CLI] tool.

You can clone the repository with:

```
git clone https://github.com/Algorithmiq/src-method.git
```

We recommend using a Git credential manager, such as [GitHub CLI], configured to
use HTTPS as protocol for Git operations.

Once the code is locally available, you can open its containing folder in
[Visual Studio Code]. The editor will then set up the [DevContainer] for you.
The first time you open the folder the startup will take a few minutes. Once the
process is done, you will have _all_ project dependencies installed, including
the git hooks.
[Visual Studio Code] will be already configured with all the extensions helpful for Python development.

**Note** that the order in which Visual Studio Code loads the extensions in the
DevContainer is non-deterministic.  You might have to execute the *Reload
Window* command to get everything to work as expected after a fresh build of the
container.

[DevContainer]: https://containers.dev/
[Docker]: https://docs.docker.com/get-docker/
[Visual Studio Code]: https://code.visualstudio.com/
[GitHub CLI]: https://cli.github.com/
[at this link]: https://docs.algorithmiq.fi/src_method
