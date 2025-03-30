'''I am doing a daily challange in https://www.deep-ml.com/'''

'''1.Matrix-Vector Dot Product

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
'''2.Transpose of a Matrix

Write a Python function that computes the transpose of a given matrix.

Example:
Input:
a = [[1,2,3],[4,5,6]]
Output:
[[1,4],[2,5],[3,6]]
Reasoning:
The transpose of a matrix is obtained by flipping rows and columns.'''


from typing import List, Union

def transpose_matrix(a: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]:
    if not a or not a[0]:  
        return []
    
    b = []  
    for col in range(len(a[0])):  
        new_row = []
        for row in range(len(a)):  
            new_row.append(a[row][col])  
        b.append(new_row) 
    
    return b




'''Reshape Matrix

Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list [ ]

Example:
Input:
a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
Output:
[[1, 2], [3, 4], [5, 6], [7, 8]]'''



import numpy as np
from typing import List, Tuple, Union

def reshape_matrix(a: List[List[Union[int, float]]], new_shape: Tuple[int, int]) -> List[List[Union[int, float]]]:
    flat_list = [num for row in a for num in row]
    total_elements = len(flat_list)
    if total_elements != new_shape[0] * new_shape[1]:
        return []  
    reshaped_matrix = np.array(flat_list).reshape(new_shape).tolist()
    return reshaped_matrix


'''Calculate Mean by Row or Column

Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

Example:
Input:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
Output:
[4.0, 5.0, 6.0]
Reasoning:
Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].'''


from typing import List

def calculate_matrix_mean(matrix: List[List[float]], mode: str) -> List[float]:
    if not matrix or not matrix[0]:  # Handle empty matrix
        return []

    if mode == 'row':  # Calculate mean for each row
        return [sum(row) / len(row) for row in matrix]

    elif mode == 'column':  # Calculate mean for each column
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        return [sum(matrix[row][col] for row in range(num_rows)) / num_rows for col in range(num_cols)]

    else:  # Handle invalid mode
        raise ValueError("Mode must be either 'row' or 'column'")



'''Scalar Multiplication of a Matrix

Write a Python function that multiplies a matrix by a scalar and returns the result.

Example:
Input:
matrix = [[1, 2], [3, 4]], scalar = 2
Output:
[[2, 4], [6, 8]]
Reasoning:
Each element of the matrix is multiplied by the scalar.'''


def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    result = [[scalar * element for element in row] for row in matrix]
	return result

'''Calculate Eigenvalues of a Matrix

Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

Example:
Input:
matrix = [[2, 1], [1, 2]]
Output:
[3.0, 1.0]
Reasoning:
The eigenvalues of the matrix are calculated using the characteristic equation of the matrix, which for a 2x2 matrix is 
λ
2
−
t
r
a
c
e
(
A
)
λ
+
d
e
t
(
A
)
=
0
λ 
2
 −trace(A)λ+det(A)=0, where 
λ
λ are the eigenvalues.'''


import numpy as np
from typing import List, Union

def calculate_eigenvalues(matrix: List[List[Union[int, float]]]) -> List[float]:
    A = np.array(matrix)
    eigenvalues = np.linalg.eigvals(A)
    return sorted(eigenvalues, reverse=True)
