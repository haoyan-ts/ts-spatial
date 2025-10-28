# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-24

### Added
- Initial release of ts-spatial as a standalone package
- `Transform3D` class for 3D spatial transformations
  - Support for multiple initialization formats (rotation matrix, quaternions, homogeneous matrix)
  - Transform composition via multiplication operator
  - Inverse transformation computation
  - Conversion to/from Pinocchio SE3 (optional)
  - Conversion to/from Viser SE3 (optional)
  - Properties for accessing position, rotation matrix, quaternions, and homogeneous matrix
  - Factory methods: `identity()`, `from_matrix()`, `from_pin_SE3()`, `from_json_node()`, `from_translation_rpy()`
- `rigid_transform()` function for computing optimal rigid transformations from point correspondences
  - Support for 2D and 3D point sets
  - Optional scale calculation for similarity transforms
  - Robust error handling with custom exceptions
- Custom exceptions for error handling:
  - `SrcDstSizeMismatchError`
  - `InvalidPointDimError`
  - `NotEnoughPointsError`
  - `RankDeficiencyError`
- Comprehensive test suite with 559 lines of unit tests
- Full type hints for better IDE integration
- Documentation with usage examples and API reference

### Changed
- Extracted from `pin_ik.spatial` module in the pin-ik project
- Made Pinocchio and Viser dependencies optional
- Added graceful fallback when optional dependencies are not installed

### Notes
- This package was created by extracting the spatial transformation functionality from the [pin-ik](https://github.com/haoyan-ts/pin-ik) project
- Maintains full backward compatibility with the original `pin_ik.spatial.transform` API
- Designed to be lightweight with minimal required dependencies (only numpy and scipy)

[0.1.0]: https://github.com/haoyan-ts/ts-spatial/releases/tag/v0.1.0
