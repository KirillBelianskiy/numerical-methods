from copy import deepcopy
from math import sqrt
import numpy as np

from matrix_processing import Matrix, eye, multiply, pmprint, read_dense_matrix

type Eigenvalue = float | complex


def qr_decomposition(
        matrix: Matrix
) -> tuple[Matrix, Matrix]:

    n = len(matrix)
    Q = eye(n)
    R = deepcopy(matrix)

    for i in range(n - 1):
        v = [0] * n
        for j in range(i, n):
            v[j] = R[j][i]

        column_norm = sqrt(sum(R[j][i] ** 2 for j in range(i, n)))

        if column_norm < 1e-6:
            continue

        sign = 1 if R[i][i] >= 0 else -1
        v[i] += sign * column_norm
        v_square_norm = sum(x ** 2 for x in v)

        H = eye(n)
        for j in range(n):
            for k in range(n):
                H[j][k] -= 2 * v[j] * v[k] / v_square_norm

        R = multiply(H, R)
        Q = multiply(Q, H)

    return Q, R


def column_tail_norm(
        matrix: Matrix,
        column: int
) -> float:
    return sqrt(sum(
        matrix[row][column] ** 2
        for row in range(column + 1, len(matrix))
    ))


def block_tail_norm(
        matrix: Matrix,
        index: int
) -> float:
    return sqrt(sum(
        matrix[row][column] ** 2
        for row in range(index + 2, len(matrix))
        for column in (index, index + 1)
    ))


def find_2x2_eigenvalues(
        matrix: Matrix,
        index: int
) -> tuple[Eigenvalue, Eigenvalue]:
    a = matrix[index][index]
    b = matrix[index][index + 1]
    c = matrix[index + 1][index]
    d = matrix[index + 1][index + 1]

    trace = a + d
    determinant = a * d - b * c
    discriminant = trace ** 2 - 4 * determinant

    if discriminant >= 0:
        root = sqrt(discriminant)
        return (trace + root) / 2, (trace - root) / 2

    real_part = trace / 2
    imaginary_part = sqrt(-discriminant) / 2
    return (
        complex(real_part, imaginary_part),
        complex(real_part, -imaginary_part)
    )


def is_qr_converged(
        matrix: Matrix,
        previous_matrix: Matrix | None,
        eps: float
) -> bool:
    i = 0
    while i < len(matrix):
        if column_tail_norm(matrix, i) <= eps:
            i += 1
            continue

        if i + 1 >= len(matrix) or block_tail_norm(matrix, i) > eps:
            return False

        first, second = find_2x2_eigenvalues(matrix, i)
        if not isinstance(first, complex) or previous_matrix is None:
            return False

        previous_first, previous_second = find_2x2_eigenvalues(previous_matrix, i)
        if max(
                abs(first - previous_first),
                abs(second - previous_second)
        ) > eps:
            return False

        i += 2

    return True


def extract_eigenvalues(
        matrix: Matrix,
        eps: float
) -> list[Eigenvalue]:
    eigenvalues = []
    i = 0

    while i < len(matrix):
        if column_tail_norm(matrix, i) <= eps:
            eigenvalues.append(matrix[i][i])
            i += 1
        else:
            first, second = find_2x2_eigenvalues(matrix, i)
            eigenvalues.extend([first, second])
            i += 2

    return eigenvalues


def find_eigenvalues(
        matrix: Matrix,
        eps: float,
        max_iterations: int
) -> tuple[list[Eigenvalue], int]:
    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be non-empty and square")

    A = deepcopy(matrix)
    previous_A = None
    iterations = 0

    while not is_qr_converged(A, previous_A, eps):
        if iterations >= max_iterations:
            raise ValueError("QR algorithm did not converge")

        previous_A = deepcopy(A)
        Q, R = qr_decomposition(A)
        A = multiply(R, Q)
        iterations += 1

    return extract_eigenvalues(A, eps), iterations


def main() -> None:
    matrix = read_dense_matrix('../input/task5.txt')

    Q, R = qr_decomposition(matrix)
    Q_R = multiply(Q, R)

    eps = 0.01
    eigenvalues, iterations = find_eigenvalues(matrix, eps, 10000)

    print(f"A:\n{pmprint(matrix)}")
    print(f"Q:\n{pmprint(Q)}")
    print(f"R:\n{pmprint(R)}")
    print(f"Q * R:\n{pmprint(Q_R)}")
    print("Eigenvalues:", eigenvalues)
    print("QR iterations:", iterations)

    complex_matrix = [
        [1.0, -1.0],
        [1.0, 1.0]
    ]
    complex_eigenvalues, _ = find_eigenvalues(complex_matrix, eps, 10000)

    numpy_eigenvalues = np.linalg.eigvals(np.array(complex_matrix))

    print(f"Complex eigenvalues example:\n{pmprint(complex_matrix)}")
    print("Calculated eigenvalues:", complex_eigenvalues)
    print("NumPy check:", numpy_eigenvalues)


if __name__ == '__main__':
    main()
