import numpy as np

eigenvalue_floor = 1e-10

#errors and validation
matrix = np.asarray(A, dtype=np.float64)
if matrix.ndim != 2:
        raise ValueError(
            name + " Must be two-dimensional, wrong matrix shape. "

        )

if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            name + " Must be square, wrong matrix shape. "

        )

if not np.all(np.isfinite(matrix)):
        raise ValueError(
            name + " Contains non applicable or infinite values."
        )

    return matrix

#Validation / aggregation

