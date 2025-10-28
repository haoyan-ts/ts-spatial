# ts-spatial

A lightweight Python library for 3D spatial transformations, providing intuitive tools for working with positions, rotations, and coordinate frame transformations.

## ✨ Features

- **Transform3D Class**: Elegant representation of 3D transformations with position and rotation
- **Multiple Input Formats**: Support for rotation matrices, quaternions (scalar-first and scalar-last), and homogeneous matrices
- **Transform Composition**: Intuitive multiplication operator for combining transforms
- **Inverse Transforms**: Efficient inverse transformation computation
- **Rigid Transform Solver**: Optimal rigid body transformation calculation from point correspondences
- **Framework Integrations**: Seamless conversion to/from Pinocchio SE3 and Viser SE3 (optional dependencies)
- **Type Hints**: Full type annotation support for better IDE integration
- **Comprehensive Tests**: 559 lines of unit tests ensuring correctness

## 📦 Installation

### Basic Installation

```bash
pip install ts-spatial
```

### With Robotics Support

For Pinocchio SE3 support (recommended for robotics applications):

```bash
# Install pinocchio via conda-forge (recommended)
conda install -c conda-forge pinocchio

# Install ts-spatial with robotics extras
pip install ts-spatial[robotics]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/haoyan-ts/ts-spatial.git
cd ts-spatial

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Creating Transforms

```python
import numpy as np
from ts_spatial import Transform3D

# Create from position and quaternion (scalar-first: w, x, y, z)
position = np.array([1.0, 2.0, 3.0])
w_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
transform = Transform3D(position=position, w_quat=w_quat)

# Create from homogeneous matrix
matrix = np.eye(4)
matrix[:3, 3] = [1.0, 2.0, 3.0]
transform = Transform3D(homogeneous=matrix)

# Create identity transform
identity = Transform3D.identity()

# Create from rotation matrix
rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° around z
transform = Transform3D(position=np.zeros(3), rotation_matrix=rotation)
```

### Transform Operations

```python
# Compose transforms
t1 = Transform3D(position=np.array([1.0, 0.0, 0.0]))
t2 = Transform3D(position=np.array([0.0, 1.0, 0.0]))
combined = t1 * t2  # Result: position=[1.0, 1.0, 0.0]

# Inverse transform
t_inv = transform.inverse()
identity = transform * t_inv  # Should equal identity

# Get difference between transforms
diff = transform1.diff(transform2)  # Returns: inv(transform1) * transform2
```

### Accessing Transform Data

```python
# Get as homogeneous matrix (4x4)
matrix = transform.homogeneous
# or
matrix = transform.to_matrix()

# Get rotation matrix (3x3)
rotation = transform.rotation_matrix

# Get quaternions
w_quat = transform.w_quat  # Scalar-first: [w, x, y, z]
quat_w = transform.quat_w  # Scalar-last: [x, y, z, w]

# Get position
position = transform.position  # Shape: (3,)

# Get as 7D vector [x, y, z, qx, qy, qz, qw]
xyzquat = transform.to_xyzquat(scalar_first=False)
```

### Rigid Transform Solver

```python
from ts_spatial import rigid_transform

# Find optimal rigid transformation from source to destination points
src_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
dst_points = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1]])

R, t, scale = rigid_transform(src_points, dst_points, calc_scale=False)

# R: 3x3 rotation matrix
# t: 3x1 translation vector
# scale: scalar (1.0 if calc_scale=False)
```

### Framework Integrations

#### Pinocchio Integration

```python
import pinocchio as pin

# Convert to Pinocchio SE3
pin_se3 = transform.to_pin_SE3()

# Create from Pinocchio SE3
transform = Transform3D.from_pin_SE3(pin_se3)
```

#### Viser Integration

```python
# Convert to Viser SE3
viser_se3 = transform.to_viser_SE3()
```

## 📚 API Reference

### Transform3D Class

#### Properties

- `position`: 3D position vector (numpy array)
- `rotation_matrix`: 3x3 rotation matrix (numpy array)
- `w_quat`: Quaternion in scalar-first format [w, x, y, z]
- `quat_w`: Quaternion in scalar-last format [x, y, z, w]
- `homogeneous`: 4x4 homogeneous transformation matrix

#### Initialization Methods

```python
Transform3D(position, rotation_matrix)
Transform3D(position, w_quat)
Transform3D(position, quat_w)
Transform3D(homogeneous)
Transform3D.identity()
Transform3D.from_matrix(matrix)
Transform3D.from_pin_SE3(se3)
Transform3D.from_json_node(json_data)
Transform3D.from_translation_rpy(translation, rpy)
```

#### Operations

```python
t1 * t2                    # Compose transforms
transform.inverse()        # Compute inverse
transform.diff(other)      # Get relative transform
transform.rebase(local)    # Apply local transform
```

#### Conversions

```python
transform.to_matrix()      # 4x4 matrix
transform.to_pin_SE3()     # Pinocchio SE3
transform.to_viser_SE3()   # Viser SE3
transform.to_xyzquat()     # 7D vector
```

### rigid_transform Function

```python
rigid_transform(src_pts, dst_pts, calc_scale=False)
```

Calculates the optimal rigid transformation from source to destination points.

**Parameters:**
- `src_pts`: Nx3 or Nx2 array of source points
- `dst_pts`: Nx3 or Nx2 array of destination points
- `calc_scale`: If True, solve for similarity transform (includes scale)

**Returns:**
- `R`: Rotation matrix (3x3 or 2x2)
- `t`: Translation vector (3x1 or 2x1)
- `scale`: Scale factor (float)

### Exceptions

- `SrcDstSizeMismatchError`: Source and destination point sets have different sizes
- `InvalidPointDimError`: Points have invalid dimensions (not 2D or 3D)
- `NotEnoughPointsError`: Insufficient points for transformation calculation
- `RankDeficiencyError`: Point set has insufficient rank

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ts_spatial --cov-report=html

# Run specific test file
pytest tests/test_transform.py
```

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black ts_spatial tests

# Type checking
mypy ts_spatial
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📖 Related Projects

- [pin-ik](https://github.com/haoyan-ts/pin-ik) - Robot wrapper class for Pinocchio-based robotics applications
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) - Fast and flexible rigid body dynamics library
- [Viser](https://github.com/nerfstudio-project/viser) - 3D visualization library

## 🔖 Citation

If you use this library in your research, please cite:

```bibtex
@software{ts_spatial,
  author = {Li, Haoyan},
  title = {ts-spatial: A lightweight Python library for 3D spatial transformations},
  year = {2025},
  url = {https://github.com/haoyan-ts/ts-spatial}
}
```

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This package was extracted from the [pin-ik](https://github.com/haoyan-ts/pin-ik) project to provide a standalone, lightweight solution for spatial transformations.
