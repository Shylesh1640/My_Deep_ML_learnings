'''Matrix-Vector Dot Product

Write a Python function that computes the dot product of a matrix and a vector. The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. For example, an n x m matrix requires a vector of length m.

Example:
Input:
a = [[1, 2], [2, 4]], b = [1, 2]
Output:
[5, 10]
Reasoning:
Row 1: (1 * 1) + (2 * 2) = 1 + 4 = 5; Row 2: (1 * 2) + (2 * 4) = 2 + 8 = 10'''


from typing import List, Union

def matrix_dot_vector(a: List[List[Union[int, float]]], b: List[Union[int, float]]) -> Union[List[Union[int, float]], int]:
    if not a or not b:
        return -1  # Handle empty matrix or vector

    num_cols = len(a[0])  # Number of columns in the matrix
    if any(len(row) != num_cols for row in a) or num_cols != len(b):
        return -1  # Check if all rows have the same length and match vector length

    return [sum(a[row][col] * b[col] for col in range(num_cols)) for row in range(len(a))]
