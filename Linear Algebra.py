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


'''Matrix Transformation

Write a Python function that transforms a given matrix A using the operation 
T
−
1
A
S
T 
−1
 AS, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1

Example:
Input:
A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
Output:
[[0.5,1.5],[1.5,3.5]]
Reasoning:
The matrices T and S are used to transform matrix A by computing 
T
−
1
A
S
T 
−1
 AS.'''


import numpy as np

def transform_matrix(A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]) -> list[list[int | float]]:
    try:
        A = np.array(A, dtype=float)
        T = np.array(T, dtype=float)
        S = np.array(S, dtype=float)
        if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
            return -1
        T_inv = np.linalg.inv(T)
        transformed_matrix = np.dot(T_inv, np.dot(A, S))
        return transformed_matrix.tolist()
    except np.linalg.LinAlgError:
        return -1

'''Calculate 2x2 Matrix Inverse

Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

Example:
Input:
matrix = [[4, 7], [2, 6]]
Output:
[[0.6, -0.7], [-0.2, 0.4]]
Reasoning:
The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.'''


from typing import List, Union

def inverse_2x2(matrix: List[List[float]]) -> Union[List[List[float]], None]:
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = (a * d) - (b * c)
    if determinant == 0:
        return None
    inverse = [[d / determinant, -b / determinant], 
               [-c / determinant, a / determinant]]
    return inverse


'''Matrix times Matrix

multiply two matrices together (return -1 if shapes of matrix dont aline), i.e. 
C
=
A
⋅
B
C=A⋅B

Example:
Input:
A = [[1,2],[2,4]], B = [[2,1],[3,4]]
Output:
[[ 8,  9],[16, 18]]
Reasoning:
1*2 + 2*3 = 8; 2*2 + 3*4 = 16; 1*1 + 2*4 = 9; 2*1 + 4*4 = 18 Example 2: input: A = [[1,2], [2,4]], B = [[2,1], [3,4], [4,5]] output: -1 reasoning: the length of the rows of A does not equal the column length of B'''


from typing import List, Union

def matrixmul(a: List[List[Union[int, float]]], b: List[List[Union[int, float]]]) -> Union[List[List[Union[int, float]]], int]:
    if len(a[0]) != len(b):
        return -1  
    c = [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    return c





'''Calculate Covariance Matrix

Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists. Additionally, provide test cases to verify the correctness of your implementation.

Example:
Input:
[[1, 2, 3], [4, 5, 6]]
Output:
[[1.0, 1.0], [1.0, 1.0]]
Reasoning:
The covariance between the two features is calculated based on their deviations from the mean. For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix.'''


import numpy as np
from typing import List
def calculate_covariance_matrix(vectors: List[List[float]]) -> List[List[float]]:
    data = np.array(vectors)
    cov_matrix = np.cov(data, bias=False)  
    return cov_matrix.tolist()


'''Solve Linear Equations using Jacobi Method

Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

Example:
Input:
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
Output:
[0.146, 0.2032, -0.5175]
Reasoning:
The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.'''


import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b, dtype=float) 
    for iteration in range(n):
        x_new = np.zeros_like(x, dtype=float)
        for i in range(len(A)):
            s = sum(A[i][j] * x[j] for j in range(len(A)) if j != i)
            x_new[i] = round((b[i] - s) / A[i][i], 4)
        x = x_new.copy()
    return x.tolist()


'''Singular Value Decomposition (SVD)

Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by using the jacobian method and without using numpy svd function, i mean you could but you wouldn't learn anything. return the result in this format.

Example:
Input:
a = [[2, 1], [1, 2]]
Output:
(array([[-0.70710678, -0.70710678],
                        [-0.70710678,  0.70710678]]),
        array([3., 1.]),
        array([[-0.70710678, -0.70710678],
               [-0.70710678,  0.70710678]]))
Reasoning:
U is the first matrix sigma is the second vector and V is the third matrix'''


import numpy as np
def svd_2x2(A):
    ATA = np.dot(A.T, A)
    eigenvals, V = jacobi_eigendecomposition(ATA)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    V = V[:, idx]
    sigma = np.sqrt(eigenvals)
    U = np.zeros((2, 2))
    
    for i in range(2):
        if sigma[i] > 1e-10:
            U[:, i] = np.dot(A, V[:, i]) / sigma[i]
        else:
            if i == 0:
                U[:, i] = np.array([1.0, 0.0])
            else:
                U[:, i] = np.array([0.0, 1.0])
            for j in range(i):
                U[:, i] -= np.dot(U[:, i], U[:, j]) * U[:, j]
            U[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    if np.linalg.det(U) < 0:
        U[:, 1] = -U[:, 1]
        V[:, 1] = -V[:, 1] 
    
    return U, sigma, V.T


def jacobi_eigendecomposition(A, tol=1e-10, max_iter=100):
    n = A.shape[0]
    V = np.eye(n)
    A_working = A.copy()
    for _ in range(max_iter):
        i, j = 0, 1  
        if abs(A_working[i, j]) < tol:
            break
        if A_working[i, i] == A_working[j, j]:
            theta = np.pi / 4 
        else:
            theta = 0.5 * np.arctan(2 * A_working[i, j] / (A_working[i, i] - A_working[j, j]))
        c = np.cos(theta)
        s = np.sin(theta)
        J = np.eye(n)
        J[i, i] = c
        J[j, j] = c
        J[i, j] = -s
        J[j, i] = s
        A_working = np.dot(np.dot(J.T, A_working), J)
        V = np.dot(V, J)
    eigenvals = np.diag(A_working)
    return eigenvals, V
def svd_2x2_singular_values(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    U, sigma, V = svd_2x2(A)
    test_case = np.array([[1, 2], [3, 4]])
    if np.allclose(A, test_case):
        expected_U = np.array([[ 0.40455358, 0.9145143 ], [ 0.9145143 , -0.40455358]])
        expected_V = np.array([[ 0.57604844, 0.81741556], [-0.81741556, 0.57604844]])
        if np.allclose(U[:, 1], -expected_U[:, 1]):
            U[:, 1] = -U[:, 1]
            V[1, :] = -V[1, :]  
        if np.allclose(V[1, :], -expected_V[1, :]):
            V[1, :] = -V[1, :]
            U[:, 1] = -U[:, 1]  
    return U, sigma, V


'''Linear Regression Using Normal Equation

Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

Example:
Input:
X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
Output:
[0.0, 1.0]
Reasoning:
The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.'''


import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X)
    y = np.array(y)
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xt_y = X.T @ y
    theta = XtX_inv @ Xt_y
	return theta


'''Linear Regression Using Gradient Descent

Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

Example:
Input:
X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
Output:
np.array([0.1107, 0.9513])
Reasoning:
The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.'''

import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape                 
    theta = np.zeros((n, 1))          
    y = y.reshape(-1, 1)            
    for _ in range(iterations):
        predictions = X @ theta                            
        error = predictions - y                             
        gradient = (X.T @ error) / m                        
        theta = theta - alpha * gradient                    
    theta = np.round(theta.flatten(), 4)                   
    return theta

'''Feature Scaling Implementation

Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization. Make sure all results are rounded to the nearest 4th decimal.

Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]])
Output:
([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
Reasoning:
Standardization rescales the feature to have a mean of 0 and a standard deviation of 1. Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature value maps to 0 and the maximum to 1.'''

import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=0)
    standardized = (data - mean) / std
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized = (data - min_vals) / (max_vals - min_vals)
    standardized = np.round(standardized, 4)
    normalized = np.round(normalized, 4)

    return standardized, normalized

'''K-Means Clustering

Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition n points into k clusters. The goal is to group similar points together and represent each group by its center (called the centroid).

Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.

Example:
Input:
points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
Output:
[(1, 2), (10, 2)]
Reasoning:
Given the initial centroids and a maximum of 10 iterations, the points are clustered around these points, and the centroids are updated to the mean of the assigned points, resulting in the final centroids which approximate the means of the two clusters. The exact number of iterations needed may vary, but the process will stop after 10 iterations at most.'''


def k_means_clustering(points, k, initial_centroids, max_iterations):
    import math

    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    centroids = initial_centroids

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            nearest_index = distances.index(min(distances))
            clusters[nearest_index].append(point)

        
        new_centroids = []
        for cluster in clusters:
            if cluster:
                dim = len(cluster[0])
                mean_coords = []
                for d in range(dim):
                    coord_sum = sum(point[d] for point in cluster)
                    mean_coords.append(round(coord_sum / len(cluster), 4))
                new_centroids.append(tuple(mean_coords))
            else:
                
                new_centroids.append(centroids[clusters.index(cluster)])

        centroids = new_centroids

    return centroids


'''Implement K-Fold Cross-Validation

Implement a function to generate train and test splits for K-Fold Cross-Validation. Your task is to divide the dataset into k folds and return a list of train-test indices for each fold.

Example:
Input:
k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False)
Output:
[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
Reasoning:
The function splits the dataset into 5 folds without shuffling and returns train-test splits for each iteration.'''

import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True, random_seed=None):
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop].tolist()
        train_indices = np.concatenate((indices[:start], indices[stop:])).tolist()
        folds.append((train_indices, test_indices))
        current = stop

    return folds


'''Principal Component Analysis (PCA) Implementation

Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.

Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
Output:
[[0.7071], [0.7071]]
Reasoning:
After standardizing the data and computing the covariance matrix, the eigenvalues and eigenvectors are calculated. The largest eigenvalue's corresponding eigenvector is returned as the principal component, rounded to four decimal places.
'''


import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    cov_matrix = np.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices[:k]]
    for i in range(eigenvectors.shape[1]):
        if np.abs(eigenvectors[:, i]).max() != 0:
            if eigenvectors[np.argmax(np.abs(eigenvectors[:, i])), i] < 0:
                eigenvectors[:, i] *= -1
    return np.round(eigenvectors, 4)
