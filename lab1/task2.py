from matrix_processing import read_sparse_matrix, pmprint


def sweep(
        a: list[float],
        b: list[float],
        c: list[float],
        d: list[float]
) -> tuple[list[float], list[float], list[float]]:
    n = len(d)

    P = [0.0] * n
    Q = [0.0] * n
    x = [0.0] * n

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denominator = b[i] + a[i - 1] * P[i - 1]

        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i - 1] * Q[i - 1]) / denominator

    P[-1] = 0

    denominator = b[-1] + a[-1] * P[-2]

    Q[-1] = (d[-1] - a[-1] * Q[-2]) / denominator
    x[-1] = Q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return P, Q, x


def main() -> None:
    a, b, c, d = read_sparse_matrix('../input/task2.txt')
    P, Q, x = sweep(a, b, c, d)
    print("P:", P)
    print("Q:", Q)
    print("x:", x)


if __name__ == "__main__":
    main()
