#!/usr/bin/env python3
"""
Module for calculating the shape of a NumPy array.

np_shape function calculates the shape of a NumPy array.
"""

def np_shape(matrix):
    """
    Calculates the shape of a numpy array.
    Args:
        matrix (numpy.ndarray): Input array of any dimensionality.
    Returns:
        tuple: A tuple representing the shape of the array.
    """
    return tuple(matrix.shape)
