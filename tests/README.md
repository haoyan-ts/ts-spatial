# Spatial Transform Tests

This directory contains unit tests for the `ts_spatial` package.

## Test Files

### `test_transform.py`

Comprehensive unit tests for the `Transform3D` class.

#### Test Coverage

The test suite covers all methods and edge cases:

**Basic Functionality:**
- ✅ `test_instantiation` - Test basic object creation
- ✅ `test_identity` - Test identity transform creation
- ✅ `test_from_json_node` - Test JSON deserialization

**Identity Transform Properties:**
- ✅ `test_identity_multiplication_left` - Test identity * transform = transform
- ✅ `test_identity_multiplication_right` - Test transform * identity = transform

**Matrix Conversion (`to_matrix`):**
- ✅ `test_to_matrix_identity` - Test identity matrix conversion
- ✅ `test_to_matrix_translation_only` - Test pure translation matrix
- ✅ `test_to_matrix_rotation_only` - Test pure rotation matrix
- ✅ `test_to_matrix_combined` - Test combined rotation and translation

**Transform Multiplication (`__mul__`):**
- ✅ `test_multiplication_two_translations` - Test combining translations
- ✅ `test_multiplication_with_rotation` - Test rotation affecting position
- ✅ `test_multiplication_associativity` - Test (A * B) * C = A * (B * C)
- ✅ `test_multiplication_matches_matrix` - Verify consistency with matrix operations

**Inverse Transform (`inverse`):**
- ✅ `test_inverse_identity` - Test identity inverse
- ✅ `test_inverse_translation` - Test pure translation inverse
- ✅ `test_inverse_rotation` - Test pure rotation inverse
- ✅ `test_inverse_left_cancellation` - Test inverse * transform = identity
- ✅ `test_inverse_right_cancellation` - Test transform * inverse = identity
- ✅ `test_double_inverse` - Test (transform⁻¹)⁻¹ = transform
- ✅ `test_inverse_matches_matrix_inverse` - Verify consistency with matrix inverse

**Pinocchio SE3 Conversion (`to_pin_SE3`):**
- ✅ `test_to_pin_SE3_identity` - Test identity conversion to Pinocchio SE3
- ✅ `test_to_pin_SE3_translation_only` - Test pure translation conversion
- ✅ `test_to_pin_SE3_rotation_only` - Test pure rotation conversion
- ✅ `test_to_pin_SE3_combined` - Test combined rotation and translation
- ✅ `test_to_pin_SE3_matches_to_matrix` - Verify consistency with to_matrix()

**Viser SE3 Conversion (`to_viser_SE3`):**
- ✅ `test_to_viser_SE3_identity` - Test identity conversion to Viser SE3
- ✅ `test_to_viser_SE3_translation_only` - Test pure translation conversion
- ✅ `test_to_viser_SE3_rotation_only` - Test pure rotation conversion
- ✅ `test_to_viser_SE3_combined` - Test combined rotation and translation
- ✅ `test_to_viser_SE3_matches_to_matrix` - Verify consistency with to_matrix()
- ✅ `test_conversion_consistency` - Test all conversion methods produce consistent results

**Edge Cases:**
- ✅ `test_edge_case_180_degree_rotation` - Test 180-degree rotations
- ✅ `test_edge_case_zero_translation` - Test transforms with no translation
- ✅ `test_quaternion_normalization_tolerance` - Test slightly unnormalized quaternions
- ✅ `test_numerical_precision` - Test numerical stability with multiple operations

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ts_spatial --cov-report=html

# Run specific test file
pytest tests/test_transform.py

# Run specific test
pytest tests/test_transform.py::TestTransform3D::test_identity
```

## Dependencies

The tests require:
- `pytest` - Test framework
- `numpy` - Numerical operations
- `scipy` - Rotation class (scipy.spatial.transform.Rotation)
- `pinocchio` - For SE3 conversion tests (optional: install via conda-forge)
- `viser` - For Viser SE3 conversion tests (optional: install via pip)
