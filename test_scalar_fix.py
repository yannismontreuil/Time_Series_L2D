#!/usr/bin/env python3
"""
Test script to verify the scalar extraction fix for matrix operations.
"""

import numpy as np

def extract_scalar(pred_val):
    """Test function that mimics our fix for scalar extraction."""
    try:
        if hasattr(pred_val, 'item'):
            return pred_val.item()
        elif pred_val.ndim == 0:
            return float(pred_val)
        elif pred_val.ndim == 1:
            return float(pred_val[0])
        else:  # matrix case
            return float(pred_val.flat[0])
    except (ValueError, IndexError) as e:
        # Fallback: try to extract any scalar value
        return float(np.asarray(pred_val).flatten()[0])

def test_scalar_extraction():
    """Test different cases of scalar extraction."""
    print("Testing scalar extraction for different numpy array types...")

    # Test case 1: 0-dimensional array (scalar)
    scalar_array = np.array(5.0)
    result1 = extract_scalar(scalar_array)
    print(f"0D array {scalar_array} -> {result1}")
    assert isinstance(result1, float)

    # Test case 2: 1D array with single element
    array_1d = np.array([5.0])
    result2 = extract_scalar(array_1d)
    print(f"1D array {array_1d} -> {result2}")
    assert isinstance(result2, float)

    # Test case 3: 2D array (1x1 matrix)
    matrix_1x1 = np.array([[5.0]])
    result3 = extract_scalar(matrix_1x1)
    print(f"1x1 matrix {matrix_1x1} -> {result3}")
    assert isinstance(result3, float)

    # Test case 4: Matrix multiplication result (simulates H @ m)
    H = np.array([[1.0, 2.0]])  # 1x2 observation matrix
    m = np.array([3.0, 4.0])    # 2x1 state vector
    pred_val = H @ m            # Results in 1D array with single element
    result4 = extract_scalar(pred_val)
    print(f"H @ m = {pred_val} (shape {pred_val.shape}) -> {result4}")
    assert isinstance(result4, float)
    assert abs(result4 - 11.0) < 1e-10  # 1*3 + 2*4 = 11

    # Test case 5: Another matrix multiplication (H @ P @ H.T)
    H = np.array([[1.0, 2.0]]).reshape(1, 2)  # 1x2 matrix
    P = np.array([[1.0, 0.5], [0.5, 2.0]])   # 2x2 covariance matrix
    result_matrix = H @ P @ H.T               # 1x1 result
    result5 = extract_scalar(result_matrix)
    print(f"H @ P @ H.T = {result_matrix} (shape {result_matrix.shape}) -> {result5}")
    assert isinstance(result5, float)

    print("\n✓ All scalar extraction tests passed!")

    # Test the innovation calculation that was causing issues
    print("\nTesting innovation calculation...")
    y_obs = 10.0
    H = np.array([[1.0, 0.5]])
    m = np.array([2.0, 4.0])
    pred_val = H @ m
    pred_scalar = extract_scalar(pred_val)
    innovation = float(y_obs - pred_scalar)
    print(f"Innovation: y={y_obs} - H@m={pred_scalar} = {innovation}")
    assert isinstance(innovation, float)

    print("✓ Innovation calculation test passed!")

if __name__ == "__main__":
    test_scalar_extraction()
