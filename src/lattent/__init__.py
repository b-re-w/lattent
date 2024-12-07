import os
import importlib
from typing import Optional


# List of supported frameworks
_FRAMEWORKS = ["pytorch", "jax"]


def _get_installed_frameworks() -> list[str]:
    """
    Detects all installed frameworks.
    Currently checks for PyTorch and JAX.

    Returns:
        list[str]: List of installed framework names
    """
    installed = []

    try:
        import torch
        installed.append("pytorch")
    except ImportError:
        pass

    try:
        import jax
        installed.append("jax")
    except ImportError:
        pass

    return installed


def _select_framework(installed_frameworks: list[str]) -> Optional[str]:
    """
    Selects which framework to use based on environment variable or availability.

    Args:
        installed_frameworks: List of installed framework names

    Returns:
        Optional[str]: Selected framework name or None if no framework is available
    """
    if not installed_frameworks:
        return None

    # Check environment variable
    preferred = os.environ.get("LATTENT_FRAMEWORK", "").lower()
    if preferred and preferred in installed_frameworks:
        return preferred

    # Default to the first installed framework
    return installed_frameworks[0]


# Detect installed frameworks
_installed_frameworks = _get_installed_frameworks()
if not _installed_frameworks:
    raise ImportError(
        "No supported framework found. "
        "Please install either PyTorch or JAX."
    )

# Select framework
_framework = _select_framework(_installed_frameworks)

# If multiple frameworks are installed, inform the user about the selection
if len(_installed_frameworks) > 1:
    print(f"Multiple frameworks detected {_installed_frameworks}. "
        + f"Using {_framework}. "
        + f"Set LATTENT_FRAMEWORK environment variable to override.")

# Import framework-specific modules
if _framework == "pytorch":
    from .pytorch import *
elif _framework == "jax":
    from .jax import *

__version__ = "0.0.1"
