"""
ts-spatial: A lightweight Python library for 3D spatial transformations.

This package provides utilities for working with 3D spatial transformations,
including position and rotation representations, transform composition, and
conversion between various formats (matrices, quaternions, etc.).

Main Components:
    - Transform3D: Class for representing and manipulating 3D transformations
    - rigid_transform: Function for computing optimal rigid transformations
    - Custom exceptions for transformation errors

Example:
    >>> from ts_spatial import Transform3D
    >>> import numpy as np
    >>>
    >>> # Create a transform
    >>> pos = np.array([1.0, 2.0, 3.0])
    >>> quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    >>> transform = Transform3D(position=pos, w_quat=quat)
    >>>
    >>> # Get homogeneous matrix
    >>> matrix = transform.homogeneous
    >>>
    >>> # Compose transforms
    >>> result = transform * transform.inverse()  # Should be identity
"""

from .transform import (
    Transform3D,
    rigid_transform,
    SrcDstSizeMismatchError,
    InvalidPointDimError,
    NotEnoughPointsError,
    RankDeficiencyError,
)

__version__ = "0.1.0"
__author__ = "Haoyan Li"

__all__ = [
    # Main classes and functions
    "Transform3D",
    "rigid_transform",
    # Exceptions
    "SrcDstSizeMismatchError",
    "InvalidPointDimError",
    "NotEnoughPointsError",
    "RankDeficiencyError",
    # Metadata
    "__version__",
]
