from matrix_processing import read_dense_matrix, pmprint, Matrix, Vector
from matrix_processing import multiply, eye, is_matrices_equal
from copy import deepcopy


def find_max_non_zero_elem(
        matrix: Matrix,
        index: int
) -> int:
    max_index = index
    for i in range(index + 1, len(matrix)):
        if abs(matrix[i][index]) > abs(matrix[max_index][index]):
            max_index = i

    return max_index


def LU_decomposition(
        matrix: Matrix
) -> tuple[Matrix, Matrix, Matrix, int]:
    shape = len(matrix)

    P = eye(shape)
    L = eye(shape)

    U = deepcopy(matrix)

    swap_count = 0
    for i in range(shape):
        k = find_max_non_zero_elem(U, i)

        if U[k][i] == 0:
            raise ValueError("Matrix is degenerate")

        if k != i:
            U[i], U[k] = U[k], U[i]
            P[i], P[k] = P[k], P[i]

            L[i][:i], L[k][:i] = L[k][:i], L[i][:i]

            swap_count += 1

        for j in range(i + 1, shape):
            l = U[j][i] / U[i][i]
            L[j][i] = l

            for k in range(i, shape):
                U[j][k] -= l * U[i][k]

    return L, U, P, swap_count


def find_determinant(
        U: Matrix,
        swap_count: int
) -> float:
    det = 1

    for i in range(len(U)):
        det *= U[i][i]

    if swap_count % 2:
        det *= -1

    return det


def find_inverse_matrix(
        P: Matrix,
        L: Matrix,
        U: Matrix
) -> Matrix:
    shape = len(P)

    inverse = [[0] * shape for _ in range(shape)]

    for column in range(shape):
        e = [0] * shape
        e[column] = 1

        x = gauss(P, L, U, e)

        for row in range(shape):
            inverse[row][column] = x[row]

    return inverse


def gauss(
        P: Matrix,
        L: Matrix,
        U: Matrix,
        b: Vector
) -> list[float]:
    shape = len(P)

    P_b = multiply(P, b)

    w = [0] * shape
    for i in range(shape):
        s = 0

        for j in range(i):
            s += L[i][j] * w[j]

        w[i] = (P_b[i] - s) / L[i][i]

    x = [0] * shape
    for i in range(shape - 1, -1, -1):
        s = 0

        for j in range(i + 1, shape):
            s += U[i][j] * x[j]

        x[i] = (w[i] - s) / U[i][i]

    return x


def main() -> None:
    filename = '../input/task1.txt'
    matrix = read_dense_matrix(filename)

    b = [x[-1] for x in matrix]
    A = [x[:-1] for x in matrix]

    L, U, P, swap_count = LU_decomposition(A)

    L_U = multiply(L, U)
    x = gauss(P, L, U, b)
    inverse_A = find_inverse_matrix(P, L, U)
    det_A = find_determinant(U, swap_count)

    print(f"L:\n{pmprint(L)}")
    print(f"U:\n{pmprint(U)}")
    print(f"L * U:\n{pmprint(L_U)}")
    print(f"x:\n{x}")
    print(f"inverse_A:\n{pmprint(inverse_A)}")
    print(f"det_A:\n{det_A}")
    print(f"A * inverse_A = E: {is_matrices_equal(multiply(A, inverse_A), eye(len(A)))}")
    print(f"L * U = P * A: {is_matrices_equal(multiply(L, U), multiply(P, A))}")


if __name__ == "__main__":
    main()
