import math
import numpy as np

eigenvalue_floor = 1e-10
symmetry_tolerance = 1e-8
conditioning_limit = 1e10
strict = True


#errors and validation

def check_matrix(A, name="matrix"):

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

def check_spd(A, name="matrix"):
    #a legal covariance matrix, symmetric with every eigenvalue positive

    matrix = check_matrix(A, name)

    if not strict:
        return matrix

    gap = np.max(np.abs(matrix - matrix.T))
    if gap > symmetry_tolerance:
        raise ValueError(
            name + " Is not symmetric, largest gap is " + str(gap)
        )

    values = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    if values.min() <= 0:
        raise ValueError(
            name + " Is not positive definite, smallest eigenvalue is "
            + str(values.min())
        )

    return matrix


def check_symmetric(A, name="matrix"):
    #tangent vectors are symmetric but not positive definite
    #so they need their own looser check

    matrix = check_matrix(A, name)

    if not strict:
        return matrix

    gap = np.max(np.abs(matrix - matrix.T))
    if gap > symmetry_tolerance:
        raise ValueError(
            name + " Is not symmetric, largest gap is " + str(gap)
        )

    return matrix


def check_same_shape(matrices, name="matrices"):
    #every matrix in a list has to be the same size

    if len(matrices) == 0:
        raise ValueError(
            name + " List is empty, nothing to work with. "
        )

    size = matrices[0].shape[0]

    for i in range(len(matrices)):
        if matrices[i].ndim != 2:
            raise ValueError(
                name + " Entry " + str(i) + " is not two-dimensional. "
            )
        if matrices[i].shape[0] != size:
            raise ValueError(
                name + " Entry " + str(i) + " has the wrong size, expected "
                + str(size)
            )

    return size


def check_conditioning(A, name="matrix"):
    #warns only, does not stop the program
    #a matrix can be legal but still numerically hopeless

    values = np.linalg.eigvalsh((A + A.T) / 2.0)
    ratio = values.max() / values.min()

    if ratio > conditioning_limit:
        print("WARNING: " + name + " is badly conditioned, ratio = "
              + format(ratio, ".2e"))

    return ratio


#cleaning

def symmetrize(A):
    #rounding makes symmetric matrices slightly lopsided, this fixes it

    return (A + A.T) / 2.0


def make_spd(A, floor=eigenvalue_floor):
    #forces a matrix to be a legal covariance matrix
    #any eigenvalue below the floor gets pushed up to the floor

    matrix = symmetrize(A)
    values, vectors = np.linalg.eigh(matrix)

    new_values = []
    for i in range(len(values)):
        if values[i] < floor:
            new_values.append(floor)
        else:
            new_values.append(values[i])

    return symmetrize(vectors @ np.diag(new_values) @ vectors.T)


#matrix functions
#all of these use the same trick, apply the function to each eigenvalue
#then rebuild the matrix, no validation here because they are internal

def sqrtm(A):

    values, vectors = np.linalg.eigh(symmetrize(A))

    new_values = []
    for i in range(len(values)):
        new_values.append(math.sqrt(values[i]))

    return symmetrize(vectors @ np.diag(new_values) @ vectors.T)


def inv_sqrtm(A):

    values, vectors = np.linalg.eigh(symmetrize(A))

    new_values = []
    for i in range(len(values)):
        new_values.append(1.0 / math.sqrt(values[i]))

    return symmetrize(vectors @ np.diag(new_values) @ vectors.T)


def logm(A):

    values, vectors = np.linalg.eigh(symmetrize(A))

    new_values = []
    for i in range(len(values)):
        new_values.append(math.log(values[i]))

    return symmetrize(vectors @ np.diag(new_values) @ vectors.T)


def expm(A):

    values, vectors = np.linalg.eigh(symmetrize(A))

    new_values = []
    for i in range(len(values)):
        new_values.append(math.exp(values[i]))

    return symmetrize(vectors @ np.diag(new_values) @ vectors.T)


def powm(A, t):

    values, vectors = np.linalg.eigh(symmetrize(A))

    new_values = []
    for i in range(len(values)):
        new_values.append(values[i] ** t)

    return symmetrize(vectors @ np.diag(new_values) @ vectors.T)


#the five primitives
#these are public so they validate their inputs

def distance(A, B):
    #affine invariant distance
    #rescale B by A, log every eigenvalue, square, add, square root
    #the log is what makes this measure risk as a ratio not a gap

    A = check_spd(A, "first matrix")
    B = check_spd(B, "second matrix")

    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "Both matrices must be the same size. "
        )

    W = inv_sqrtm(A)
    M = symmetrize(W @ B @ W)
    values = np.linalg.eigvalsh(M)

    if values.min() <= 0:
        raise ValueError(
            "Rescaled matrix lost positive definiteness, inputs are too "
            "badly conditioned. "
        )

    total = 0.0
    for i in range(len(values)):
        total = total + math.log(values[i]) ** 2

    return math.sqrt(total)


def log_map(P, X):
    #turns the point X into an arrow pointing away from P
    #arrows live in flat space so we can average them normally

    P = check_spd(P, "base point")
    X = check_spd(X, "target point")

    if P.shape[0] != X.shape[0]:
        raise ValueError(
            "Base point and target must be the same size. "
        )

    S = sqrtm(P)
    W = inv_sqrtm(P)

    return symmetrize(S @ logm(symmetrize(W @ X @ W)) @ S)


def exp_map(P, V):
    #opposite of log_map, follow the arrow V from P and land somewhere
    #the landing spot is always a legal covariance matrix

    P = check_spd(P, "base point")
    V = check_symmetric(V, "tangent vector")

    if P.shape[0] != V.shape[0]:
        raise ValueError(
            "Base point and tangent vector must be the same size. "
        )

    S = sqrtm(P)
    W = inv_sqrtm(P)

    return make_spd(S @ expm(symmetrize(W @ V @ W)) @ S)


def geodesic(A, B, t):
    #the point t of the way along the shortest curved path from A to B
    #t = 0 gives A, t = 1 gives B, t above 1 goes past B

    A = check_spd(A, "start point")
    B = check_spd(B, "end point")

    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "Start and end must be the same size. "
        )

    if not np.isfinite(t):
        raise ValueError(
            "Travel fraction t must be a finite number. "
        )

    S = sqrtm(A)
    W = inv_sqrtm(A)

    return make_spd(S @ powm(symmetrize(W @ B @ W), t) @ S)


def frechet_mean(matrices, tol=1e-9, max_iter=50):
    #the average of covariance matrices done properly
    #guess a center, average the arrows to every matrix, step, repeat
    #plain adding and dividing inflates risk, this does not

    size = check_same_shape(matrices, "matrix list")

    for i in range(len(matrices)):
        check_spd(matrices[i], "matrix " + str(i))

    P = matrices[0].copy()
    for i in range(1, len(matrices)):
        P = P + matrices[i]
    P = make_spd(P / len(matrices))

    for step in range(max_iter):

        arrow_sum = np.zeros((size, size))
        for i in range(len(matrices)):
            arrow_sum = arrow_sum + log_map(P, matrices[i])
        average_arrow = arrow_sum / len(matrices)

        if np.linalg.norm(average_arrow) < tol:
            break

        P = exp_map(P, average_arrow)

    return P


def frechet_variance(matrices, center):
    #average squared distance to the center, like ordinary variance

    check_same_shape(matrices, "matrix list")
    center = check_spd(center, "center")

    total = 0.0
    for i in range(len(matrices)):
        total = total + distance(center, matrices[i]) ** 2

    return total / len(matrices)


#flattening
#machine learning wants rows of numbers, not matrices

def vectorize(V):
    #keep the upper triangle only since the matrix repeats itself
    #off diagonal entries get times root two so lengths stay honest

    V = check_symmetric(V, "matrix to flatten")

    size = V.shape[0]
    out = []

    for i in range(size):
        out.append(V[i][i])
        for j in range(i + 1, size):
            out.append(math.sqrt(2.0) * V[i][j])

    return np.array(out)


def unvectorize(v, size):
    #exact opposite of vectorize

    expected = size * (size + 1) // 2
    if len(v) != expected:
        raise ValueError(
            "Wrong number of values, expected " + str(expected)
            + " but got " + str(len(v))
        )

    V = np.zeros((size, size))
    position = 0

    for i in range(size):
        V[i][i] = v[position]
        position = position + 1
        for j in range(i + 1, size):
            value = v[position] / math.sqrt(2.0)
            V[i][j] = value
            V[j][i] = value
            position = position + 1

    return V


def tangent_features(center, matrices):
    #turns a whole list of covariance matrices into rows of numbers
    #arrow from center to each matrix, then flatten each arrow

    center = check_spd(center, "center")
    check_same_shape(matrices, "matrix list")

    rows = []
    for i in range(len(matrices)):
        rows.append(vectorize(log_map(center, matrices[i])))

    return np.array(rows)
