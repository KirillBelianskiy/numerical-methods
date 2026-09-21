from copy import deepcopy
from math import atan, sin, cos, pi

from matrix_processing import eye, T, multiply, read_dense_matrix, pmprint


def max_upper_triangular(
        matrix: list[list[float]]
) -> tuple[float, float]:
    n = len(matrix)
    max_elem = matrix[0][1]
    i_max_elem, j_max_elem = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) > abs(max_elem):
                max_elem = matrix[i][j]
                i_max_elem = i
                j_max_elem = j

    return i_max_elem, j_max_elem


def t(
        matrix: list[list[float]]
) -> float:
    n = len(matrix)

    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += matrix[i][j] ** 2

    return s ** 0.5


def jacobi(
        matrix: list[list[float]],
        eps: float,
        max_iteration: int,
) -> tuple[list[list[float | int]], list[list[float | int]]]:
    n = len(matrix)
    A = deepcopy(matrix)
    k = 0
    U_total = eye(n)
    while k < max_iteration:
        if t(A) < eps:
            break

        i, j = max_upper_triangular(A)

        if A[i][i] == A[j][j]:
            phi = pi / 4
        else:
            phi = 0.5 * atan(
                2 * A[i][j] /
                (A[i][i] - A[j][j])
            )
        c = cos(phi)
        s = sin(phi)

        U_k = eye(n)
        U_k[i][i] = c
        U_k[j][j] = c

        U_k[i][j] = -s
        U_k[j][i] = s

        A = multiply(multiply((T(U_k)), A), U_k)

        U_total = multiply(U_total, U_k)

        k += 1

    return A, U_total


def main() -> None:
    matrix = read_dense_matrix('../input/task4.txt')

    eps = 0.01
    A, U = jacobi(matrix, eps, 100000)

    print("Accuracy: ", eps)
    print("A:", pmprint(matrix))
    print("Eigenvalues:", [A[i][i] for i in range(len(A))])
    print("Eigenvectors:", pmprint(U))


if __name__ == '__main__':
    main()
