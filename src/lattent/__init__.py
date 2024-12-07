import os
import importlib
from typing import Optional


# List of supported frameworks
_FRAMEWORKS = ["pytorch", "jax"]


def _get_installed_framework() -> Optional[str]:
    """
    Detects and returns the installed framework.
    Currently, supports PyTorch and JAX.

    Returns:
        Optional[str]: Name of the detected framework or None
    """
    # Check for PyTorch
    try:
        import torch
        return "pytorch"
    except ImportError:
        pass

    # Check for JAX
    try:
        import jax
        return "jax"
    except ImportError:
        pass

    return None


def _import_framework_module(framework: str):
    """
    Imports the appropriate module for the given framework.

    Args:
        framework (str): Name of the framework ("pytorch" or "jax")
    """
    if framework not in _FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {framework}. Supported frameworks: {_FRAMEWORKS}")

    # Check if framework preference is set in environment variables
    env_framework = os.getenv("PREFERRED_FRAMEWORK")
    if env_framework and env_framework.lower() != framework:
        print(f"Warning: Mismatch between environment variable framework ({env_framework}) "
              f"and detected framework ({framework}). Using detected framework.")

    # Import framework-specific modules
    if framework == "pytorch":
        from .models.ttt_pytorch import *
    elif framework == "jax":
        from .models.ttt_layer import *
        from .models.model import *


# Detect framework and import modules
_framework = _get_installed_framework()
if _framework is None:
    raise ImportError(
        "No supported framework found. "
        "Please install either PyTorch or JAX."
    )

_import_framework_module(_framework)

__version__ = "0.0.1"
