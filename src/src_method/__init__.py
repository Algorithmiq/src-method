"""Copyright (c) 2025 Algorithmiq Development Team. All rights reserved.

src_method: Successive Randomized Compression.
"""

from __future__ import annotations

from ._version import version as __version__
from ._version import version_tuple as __version_tuple__
from .apply import apply
from .compress import compress

__all__ = ["__version__", "__version_tuple__", "apply", "compress"]
