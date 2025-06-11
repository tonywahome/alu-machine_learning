import numpy as np

A = [[-2, -4, 2],
     [-2, 1, 2],
     [4, 2, 5]]

V = [[2], [-3], [-1]]


print("Matrix A:")
print(A)
print("\nVector V:")
print(V)

# Perform matrix multiplication
result = np.dot(A, V)
R = print("\nResult of A * V:")
print(R)
# Cannot print R directly as it is None, so we print the result instead
print(result)
# Check if the result is a vector
if len(result.shape) == 1 or (len(result.shape) == 2 and result.shape[1] == 1):
    print("\nThe result is a vector.")
else:
    print("\nThe result is not a vector.")

# Check if the result is a matrix
if len(result.shape) == 2 and result.shape[0] > 1 and result.shape[1] > 1:
    print("\nThe result is a matrix.")
else:
    print("\nThe result is not a matrix.")

# Check if the result is a scalar
if len(result.shape) == 0:
    print("\nThe result is a scalar.")
else:
    print("\nThe result is not a scalar.")


