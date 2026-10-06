"""Copyright (c) 2025 Algorithmiq Development Team. All rights reserved.

src_method: Successive Randomized Compression.
"""

from __future__ import annotations

import logging

from ._version import version as __version__
from ._version import version_tuple as __version_tuple__
from .apply import apply
from .compress import compress
from .stack import src

# Stay silent unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__", "__version_tuple__", "apply", "compress", "src"]
