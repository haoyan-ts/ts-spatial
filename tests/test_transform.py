"""Unit tests for Transform3D class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import pinocchio as pin
from viser.transforms import SE3 as ViserSE3

from ts_spatial import Transform3D


class TestTransform3D:
    """Test suite for Transform3D class."""

    def test_instantiation(self):
        """Test basic instantiation of Transform3D."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion [w, x, y, z]

        transform = Transform3D(position=position, w_quat=w_quat)

        assert_array_equal(transform.position, position)
        assert_array_equal(transform.w_quat, w_quat)
        assert transform.position.shape == (3,)
        assert transform.w_quat.shape == (4,)

    def test_identity(self):
        """Test identity transform creation."""
        identity = Transform3D.identity()

        # Identity should have zero position
        assert_allclose(identity.position, np.zeros(3))

        # Identity should have unit quaternion [1, 0, 0, 0]
        expected_w_quat = np.array([1.0, 0.0, 0.0, 0.0])
        assert_allclose(identity.w_quat, expected_w_quat)

    def test_identity_multiplication_left(self):
        """Test that identity * transform = transform."""
        position = np.array([1.0, 2.0, 3.0])
        # Use properly normalized quaternion
        w_quat = np.array(
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0]
        )  # 90° rotation around x
        transform = Transform3D(position=position, w_quat=w_quat)

        identity = Transform3D.identity()
        result = identity * transform

        assert_allclose(result.position, transform.position, rtol=1e-6)
        assert_allclose(result.w_quat, transform.w_quat, rtol=1e-6)

    def test_identity_multiplication_right(self):
        """Test that transform * identity = transform."""
        position = np.array([1.0, 2.0, 3.0])
        # Use properly normalized quaternion
        w_quat = np.array(
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0]
        )  # 90° rotation around x
        transform = Transform3D(position=position, w_quat=w_quat)

        identity = Transform3D.identity()
        result = transform * identity

        assert_allclose(result.position, transform.position, rtol=1e-6)
        assert_allclose(result.w_quat, transform.w_quat, rtol=1e-6)

    def test_from_json_node(self):
        """Test creation from JSON node data."""
        json_data = {
            "position": {"x": 1.5, "y": 2.5, "z": 3.5},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }

        transform = Transform3D.from_json_node(json_data)

        expected_position = np.array([1.5, 2.5, 3.5])
        expected_w_quat = np.array([1.0, 0.0, 0.0, 0.0])

        assert_allclose(transform.position, expected_position)
        assert_allclose(transform.w_quat, expected_w_quat)

    def test_to_matrix_identity(self):
        """Test that identity transform produces identity matrix."""
        identity = Transform3D.identity()
        matrix = identity.to_matrix()

        expected_matrix = np.eye(4)
        assert_allclose(matrix, expected_matrix, rtol=1e-6)

    def test_to_matrix_translation_only(self):
        """Test matrix conversion with only translation."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
        transform = Transform3D(position=position, w_quat=w_quat)

        matrix = transform.to_matrix()

        # Check that rotation part is identity
        assert_allclose(matrix[:3, :3], np.eye(3), rtol=1e-6)

        # Check that translation is correct
        assert_allclose(matrix[:3, 3], position, rtol=1e-6)

        # Check that bottom row is [0, 0, 0, 1]
        assert_allclose(matrix[3, :], np.array([0, 0, 0, 1]), rtol=1e-6)

    def test_to_matrix_rotation_only(self):
        """Test matrix conversion with only rotation (90° around z-axis)."""
        position = np.zeros(3)
        # 90° rotation around z-axis: [w, x, y, z] = [cos(45°), 0, 0, sin(45°)]
        w_quat = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        transform = Transform3D(position=position, w_quat=w_quat)

        matrix = transform.to_matrix()

        # Expected rotation matrix for 90° around z
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        assert_allclose(matrix[:3, :3], expected_rotation, atol=1e-6)
        assert_allclose(matrix[:3, 3], np.zeros(3), atol=1e-6)

    def test_to_matrix_combined(self):
        """Test matrix conversion with both rotation and translation."""
        position = np.array([1.0, 2.0, 3.0])
        # 90° rotation around z-axis
        w_quat = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        transform = Transform3D(position=position, w_quat=w_quat)

        matrix = transform.to_matrix()

        # Check dimensions
        assert matrix.shape == (4, 4)

        # Check that the transformation is applied correctly
        # Transform a point using matrix and verify
        point = np.array([1.0, 0.0, 0.0, 1.0])  # homogeneous coordinates
        transformed = matrix @ point

        # After 90° rotation around z: (1,0,0) -> (0,1,0), then add translation
        expected = np.array([1.0, 3.0, 3.0, 1.0])
        assert_allclose(transformed, expected, atol=1e-6)

    def test_multiplication_two_translations(self):
        """Test multiplication of two pure translations."""
        t1 = Transform3D(
            position=np.array([1.0, 0.0, 0.0]), w_quat=np.array([1.0, 0.0, 0.0, 0.0])
        )
        t2 = Transform3D(
            position=np.array([0.0, 2.0, 0.0]), w_quat=np.array([1.0, 0.0, 0.0, 0.0])
        )

        result = t1 * t2

        # Translations should add
        expected_position = np.array([1.0, 2.0, 0.0])
        assert_allclose(result.position, expected_position, rtol=1e-6)

        # Rotation should remain identity
        expected_w_quat = np.array([1.0, 0.0, 0.0, 0.0])
        assert_allclose(result.w_quat, expected_w_quat, rtol=1e-6)

    def test_multiplication_with_rotation(self):
        """Test multiplication with rotation affecting position."""
        # First transform: 90° rotation around z-axis
        t1 = Transform3D(
            position=np.zeros(3),
            w_quat=np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2]),
        )

        # Second transform: translation along x
        t2 = Transform3D(
            position=np.array([1.0, 0.0, 0.0]), w_quat=np.array([1.0, 0.0, 0.0, 0.0])
        )

        result = t1 * t2

        # After 90° rotation around z, x-axis points along y
        # So translation (1,0,0) becomes (0,1,0)
        expected_position = np.array([0.0, 1.0, 0.0])
        assert_allclose(result.position, expected_position, atol=1e-6)

    def test_multiplication_associativity(self):
        """Test that (A * B) * C = A * (B * C)."""
        t1 = Transform3D(
            position=np.array([1.0, 0.0, 0.0]),
            w_quat=np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0]),
        )
        t2 = Transform3D(
            position=np.array([0.0, 1.0, 0.0]),
            w_quat=np.array([np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0]),
        )
        t3 = Transform3D(
            position=np.array([0.0, 0.0, 1.0]),
            w_quat=np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2]),
        )

        result1 = (t1 * t2) * t3
        result2 = t1 * (t2 * t3)

        assert_allclose(result1.position, result2.position, rtol=1e-6)
        assert_allclose(result1.w_quat, result2.w_quat, rtol=1e-6)

    def test_multiplication_matches_matrix(self):
        """Test that transform multiplication matches matrix multiplication."""
        t1 = Transform3D(
            position=np.array([1.0, 2.0, 3.0]),
            w_quat=np.array([0.707, 0.707, 0.0, 0.0]),
        )
        t2 = Transform3D(
            position=np.array([0.5, 1.5, 2.5]),
            w_quat=np.array([0.707, 0.0, 0.707, 0.0]),
        )

        # Using transform multiplication
        result_transform = t1 * t2
        result_matrix_from_transform = result_transform.to_matrix()

        # Using matrix multiplication
        m1 = t1.to_matrix()
        m2 = t2.to_matrix()
        result_matrix_from_matrices = m1 @ m2

        assert_allclose(
            result_matrix_from_transform, result_matrix_from_matrices, rtol=1e-6
        )

    def test_inverse_identity(self):
        """Test that identity inverse is identity."""
        identity = Transform3D.identity()
        inv = identity.inverse()

        assert_allclose(inv.position, identity.position, rtol=1e-6)
        assert_allclose(inv.w_quat, identity.w_quat, rtol=1e-6)

    def test_inverse_translation(self):
        """Test inverse of pure translation."""
        position = np.array([1.0, 2.0, 3.0])
        transform = Transform3D(
            position=position, w_quat=np.array([1.0, 0.0, 0.0, 0.0])
        )

        inv = transform.inverse()

        # Inverse translation should negate position
        assert_allclose(inv.position, -position, rtol=1e-6)
        assert_allclose(inv.w_quat, np.array([1.0, 0.0, 0.0, 0.0]), rtol=1e-6)

    def test_inverse_rotation(self):
        """Test inverse of pure rotation."""
        # 90° rotation around z-axis
        w_quat = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        transform = Transform3D(position=np.zeros(3), w_quat=w_quat)

        inv = transform.inverse()

        # Inverse rotation: -90° around z-axis
        # Note: quaternions q and -q represent the same rotation
        expected_w_quat_1 = np.array([np.sqrt(2) / 2, 0.0, 0.0, -np.sqrt(2) / 2])
        expected_w_quat_2 = np.array([-np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        assert_allclose(inv.position, np.zeros(3), atol=1e-6)
        assert np.allclose(inv.w_quat, expected_w_quat_1, atol=1e-6) or np.allclose(
            inv.w_quat, expected_w_quat_2, atol=1e-6
        )

    def test_inverse_left_cancellation(self):
        """Test that inverse * transform = identity."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([0.6, 0.8, 0.0, 0.0])  # arbitrary rotation
        # Normalize quaternion
        w_quat = w_quat / np.linalg.norm(w_quat)

        transform = Transform3D(position=position, w_quat=w_quat)
        inv = transform.inverse()

        result = inv * transform

        # Result should be close to identity
        assert_allclose(result.position, np.zeros(3), atol=1e-6)
        # Check quaternion is close to [1,0,0,0] or [-1,0,0,0] (same rotation)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        # Quaternions q and -q represent the same rotation
        assert np.allclose(result.w_quat, identity_quat, atol=1e-6) or np.allclose(
            result.w_quat, -identity_quat, atol=1e-6
        )

    def test_inverse_right_cancellation(self):
        """Test that transform * inverse = identity."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([0.5, 0.5, 0.5, 0.5])  # arbitrary rotation
        # Normalize quaternion
        w_quat = w_quat / np.linalg.norm(w_quat)

        transform = Transform3D(position=position, w_quat=w_quat)
        inv = transform.inverse()

        result = transform * inv

        # Result should be close to identity
        assert_allclose(result.position, np.zeros(3), atol=1e-6)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result.w_quat, identity_quat, atol=1e-6) or np.allclose(
            result.w_quat, -identity_quat, atol=1e-6
        )

    def test_double_inverse(self):
        """Test that (transform.inverse()).inverse() = transform."""
        position = np.array([1.0, 2.0, 3.0])
        # Use properly normalized quaternion
        w_quat = np.array([np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0])
        transform = Transform3D(position=position, w_quat=w_quat)

        double_inv = transform.inverse().inverse()

        assert_allclose(double_inv.position, transform.position, rtol=1e-6)
        # Quaternions may differ by sign
        assert np.allclose(
            double_inv.w_quat, transform.w_quat, rtol=1e-6
        ) or np.allclose(double_inv.w_quat, -transform.w_quat, rtol=1e-6)

    def test_inverse_matches_matrix_inverse(self):
        """Test that inverse transform matches matrix inverse."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([0.6, 0.4, 0.5, 0.5])
        w_quat = w_quat / np.linalg.norm(w_quat)  # normalize

        transform = Transform3D(position=position, w_quat=w_quat)

        # Get inverse using transform method
        inv_transform = transform.inverse()
        inv_matrix_from_transform = inv_transform.to_matrix()

        # Get inverse using matrix inversion
        matrix = transform.to_matrix()
        inv_matrix_from_numpy = np.linalg.inv(matrix)

        assert_allclose(inv_matrix_from_transform, inv_matrix_from_numpy, rtol=1e-6)

    def test_edge_case_180_degree_rotation(self):
        """Test 180-degree rotation around x-axis."""
        # 180° rotation: [w, x, y, z] = [0, 1, 0, 0]
        w_quat = np.array([0.0, 1.0, 0.0, 0.0])
        transform = Transform3D(position=np.zeros(3), w_quat=w_quat)

        matrix = transform.to_matrix()

        # 180° around x should flip y and z
        expected_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        assert_allclose(matrix[:3, :3], expected_rotation, atol=1e-6)

    def test_edge_case_zero_translation(self):
        """Test transform with zero translation."""
        w_quat = np.array([0.707, 0.0, 0.707, 0.0])
        transform = Transform3D(position=np.zeros(3), w_quat=w_quat)

        matrix = transform.to_matrix()

        # Translation part should be zero
        assert_allclose(matrix[:3, 3], np.zeros(3), atol=1e-6)

    def test_quaternion_normalization_tolerance(self):
        """Test that slightly unnormalized quaternions work correctly."""
        # Slightly unnormalized quaternion
        w_quat = np.array([0.71, 0.71, 0.0, 0.0])  # Should be ~0.707
        transform = Transform3D(position=np.array([1.0, 0.0, 0.0]), w_quat=w_quat)

        # Should not raise an error and produce valid matrix
        matrix = transform.to_matrix()
        assert matrix.shape == (4, 4)

        # Verify that rotation matrix is orthogonal (within tolerance)
        rot_matrix = matrix[:3, :3]
        should_be_identity = rot_matrix.T @ rot_matrix
        assert_allclose(should_be_identity, np.eye(3), atol=1e-2)

    def test_numerical_precision(self):
        """Test numerical precision with multiple operations."""
        # Create a transform and apply inverse 10 times
        w_quat = np.array([0.6, 0.4, 0.5, 0.5])
        w_quat = w_quat / np.linalg.norm(w_quat)  # normalize

        original = Transform3D(position=np.array([1.0, 2.0, 3.0]), w_quat=w_quat)

        result = original
        for _ in range(10):
            result = result * result.inverse()

        # Should still be close to identity
        identity = Transform3D.identity()
        assert_allclose(result.position, identity.position, atol=1e-5)

    def test_to_pin_SE3_identity(self):
        """Test conversion of identity transform to Pinocchio SE3."""
        identity = Transform3D.identity()
        pin_se3 = identity.to_pin_SE3()

        # Check that it's a Pinocchio SE3 object
        assert isinstance(pin_se3, pin.SE3)

        # Check rotation is identity
        assert_allclose(pin_se3.rotation, np.eye(3), atol=1e-6)

        # Check translation is zero
        assert_allclose(pin_se3.translation, np.zeros(3), atol=1e-6)

    def test_to_pin_SE3_translation_only(self):
        """Test conversion of pure translation to Pinocchio SE3."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
        transform = Transform3D(position=position, w_quat=w_quat)

        pin_se3 = transform.to_pin_SE3()

        # Check rotation is identity
        assert_allclose(pin_se3.rotation, np.eye(3), atol=1e-6)

        # Check translation matches
        assert_allclose(pin_se3.translation, position, atol=1e-6)

    def test_to_pin_SE3_rotation_only(self):
        """Test conversion of pure rotation to Pinocchio SE3."""
        position = np.zeros(3)
        # 90° rotation around z-axis
        w_quat = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        transform = Transform3D(position=position, w_quat=w_quat)

        pin_se3 = transform.to_pin_SE3()

        # Check translation is zero
        assert_allclose(pin_se3.translation, np.zeros(3), atol=1e-6)

        # Check rotation matrix (90° around z should give specific matrix)
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert_allclose(pin_se3.rotation, expected_rotation, atol=1e-6)

    def test_to_pin_SE3_combined(self):
        """Test conversion with both rotation and translation to Pinocchio SE3."""
        position = np.array([1.0, 2.0, 3.0])
        # 90° rotation around x-axis
        w_quat = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0])
        transform = Transform3D(position=position, w_quat=w_quat)

        pin_se3 = transform.to_pin_SE3()

        # Check translation
        assert_allclose(pin_se3.translation, position, atol=1e-6)

        # Check rotation matrix (90° around x)
        expected_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert_allclose(pin_se3.rotation, expected_rotation, atol=1e-6)

    def test_to_pin_SE3_matches_to_matrix(self):
        """Test that Pinocchio SE3 homogeneous matrix matches to_matrix()."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([0.6, 0.4, 0.5, 0.5])
        w_quat = w_quat / np.linalg.norm(w_quat)  # normalize
        transform = Transform3D(position=position, w_quat=w_quat)

        pin_se3 = transform.to_pin_SE3()
        matrix_from_transform = transform.to_matrix()
        matrix_from_pin = pin_se3.homogeneous

        assert_allclose(matrix_from_pin, matrix_from_transform, rtol=1e-6)

    def test_to_viser_SE3_identity(self):
        """Test conversion of identity transform to Viser SE3."""
        identity = Transform3D.identity()
        viser_se3 = identity.to_viser_SE3()

        # Check that it's a Viser SE3 object
        assert isinstance(viser_se3, ViserSE3)

        # Check that the matrix is identity
        expected_matrix = np.eye(4)
        assert_allclose(viser_se3.as_matrix(), expected_matrix, atol=1e-6)

    def test_to_viser_SE3_translation_only(self):
        """Test conversion of pure translation to Viser SE3."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
        transform = Transform3D(position=position, w_quat=w_quat)

        viser_se3 = transform.to_viser_SE3()
        matrix = viser_se3.as_matrix()

        # Check rotation part is identity
        assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-6)

        # Check translation part
        assert_allclose(matrix[:3, 3], position, atol=1e-6)

    def test_to_viser_SE3_rotation_only(self):
        """Test conversion of pure rotation to Viser SE3."""
        position = np.zeros(3)
        # 90° rotation around z-axis
        w_quat = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])
        transform = Transform3D(position=position, w_quat=w_quat)

        viser_se3 = transform.to_viser_SE3()
        matrix = viser_se3.as_matrix()

        # Check translation part is zero
        assert_allclose(matrix[:3, 3], np.zeros(3), atol=1e-6)

        # Check rotation part (90° around z)
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert_allclose(matrix[:3, :3], expected_rotation, atol=1e-6)

    def test_to_viser_SE3_combined(self):
        """Test conversion with both rotation and translation to Viser SE3."""
        position = np.array([1.0, 2.0, 3.0])
        # 90° rotation around y-axis
        w_quat = np.array([np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0])
        transform = Transform3D(position=position, w_quat=w_quat)

        viser_se3 = transform.to_viser_SE3()
        matrix = viser_se3.as_matrix()

        # Check translation
        assert_allclose(matrix[:3, 3], position, atol=1e-6)

        # Check rotation (90° around y)
        expected_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        assert_allclose(matrix[:3, :3], expected_rotation, atol=1e-6)

    def test_to_viser_SE3_matches_to_matrix(self):
        """Test that Viser SE3 matrix matches to_matrix()."""
        position = np.array([1.0, 2.0, 3.0])
        w_quat = np.array([0.6, 0.4, 0.5, 0.5])
        w_quat = w_quat / np.linalg.norm(w_quat)  # normalize
        transform = Transform3D(position=position, w_quat=w_quat)

        viser_se3 = transform.to_viser_SE3()
        matrix_from_viser = viser_se3.as_matrix()
        matrix_from_transform = transform.to_matrix()

        assert_allclose(matrix_from_viser, matrix_from_transform, rtol=1e-6)

    def test_conversion_consistency(self):
        """Test that all conversion methods produce consistent results."""
        position = np.array([1.5, 2.5, 3.5])
        w_quat = np.array([0.5, 0.5, 0.5, 0.5])
        w_quat = w_quat / np.linalg.norm(w_quat)
        transform = Transform3D(position=position, w_quat=w_quat)

        # Get matrices from all conversion methods
        matrix_direct = transform.to_matrix()
        matrix_from_pin = transform.to_pin_SE3().homogeneous
        matrix_from_viser = transform.to_viser_SE3().as_matrix()

        # All should be equal (use atol for numerical precision near zero)
        assert_allclose(matrix_from_pin, matrix_direct, rtol=1e-6, atol=1e-15)
        assert_allclose(matrix_from_viser, matrix_direct, rtol=1e-6, atol=1e-15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
