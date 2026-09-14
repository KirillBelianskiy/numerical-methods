from matrix_processing import read_dense_matrix


def fixed_point_iteration(
        A: list[list[float]],
        b: list[float],
        eps: float,
        max_iterations: int
) -> list[float]:
    n = len(b)

    x_last = [0] * n
    x_new = [0] * n
    k = 0
    while k < max_iterations:
        for i in range(n):
            x_new[i] = (b[i] - sum(A[i][j] * x_last[j] for j in range(n) if i != j)) / A[i][i]

        k += 1

        if max(abs(x_new[i] - x_last[i]) for i in range(n)) < eps:
            break

        x_last = x_new.copy()

    return x_new


def gauss_seidel(
        A: list[list[float]],
        b: list[float],
        eps: float,
        max_iterations: int
) -> list[float]:
    n = len(b)

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

        if max(abs(x_new[i] - x_last[i]) for i in range(n)) < eps:
            break

        x_last = x_new.copy()

    return x_new


def main() -> None:
    A = read_dense_matrix('../input/task3.txt')
    b = [x[-1] for x in A]
    A = [x[:-1] for x in A]

    x_fpi = fixed_point_iteration(A, b, eps=1e-6, max_iterations=100)
    x_gs = gauss_seidel(A, b, eps=1e-6, max_iterations=100)
    print(x_fpi, x_gs, sep='\n\n')


if __name__ == '__main__':
    main()
