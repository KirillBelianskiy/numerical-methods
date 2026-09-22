from matrix_processing import read_dense_matrix, Matrix, Vector


def norm(
        A: Matrix
) -> float:
    n = len(A)

    alpha = [
        [
            0 if i == j else -A[i][j] / A[i][i]
            for j in range(n)
        ]
        for i in range(n)
    ]

    return max(
        sum(abs(alpha[i][j]) for j in range(n))
        for i in range(n)
    )


def fixed_point_iteration(
        A: Matrix,
        b: Vector,
        eps: float,
        max_iterations: int
) -> tuple[Vector, int]:
    n = len(b)

    q = norm(A)

    x_last = [0] * n
    x_new = [0] * n
    k = 0
    while k < max_iterations:
        for i in range(n):
            x_new[i] = (b[i] - sum(A[i][j] * x_last[j] for j in range(n) if i != j)) / A[i][i]

        k += 1

        diff = max(abs(x_new[i] - x_last[i]) for i in range(n))

        if q < 1:
            if q / (1 - q) * diff < eps:
                break
        else:
            if diff < eps:
                break

        x_last = x_new.copy()

    return x_new, k


def gauss_seidel(
        A: Matrix,
        b: Vector,
        eps: float,
        max_iterations: int
) -> tuple[Vector, int]:
    n = len(b)

    q = norm(A)

    x_last = [0] * n
    x_new = [0] * n
    k = 0
    while k < max_iterations:
        for i in range(n):
            x_new[i] = (
                               b[i] -
                               sum(A[i][j] * x_new[j] for j in range(i)) -
                               sum(A[i][j] * x_last[j] for j in range(i + 1, n))
                       ) / A[i][i]

        k += 1

        diff = max(abs(x_new[i] - x_last[i]) for i in range(n))

        if q < 1:
            if q / (1 - q) * diff < eps:
                break
        else:
            if diff < eps:
                break

        x_last = x_new.copy()

    return x_new, k


def main() -> None:
    A = read_dense_matrix('../input/task3.txt')
    b = [x[-1] for x in A]
    A = [x[:-1] for x in A]

    eps = 0.01
    x_fpi, k_fpi = fixed_point_iteration(A, b, eps=eps, max_iterations=1000)
    x_gs, k_gs = gauss_seidel(A, b, eps=eps, max_iterations=1000)
    print("Accuracy", eps)
    print("Fixed point iterations:", k_fpi)
    print(x_fpi)
    print("Gauss-Seidel iterations:", k_gs)
    print(x_gs)
    print("Reference solution", [8, 4, 3, 9])


if __name__ == '__main__':
    main()
