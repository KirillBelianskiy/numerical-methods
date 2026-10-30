from lab1.task1 import LU_decomposition, gauss
from matrix_processing import Matrix, Vector


def normal_system(
        x_values: Vector,
        y_values: Vector,
        n: int
) -> tuple[Matrix, Vector]:
    A = [
        [sum(x ** (k + i) for x in x_values) for i in range(n + 1)]
        for k in range(n + 1)
    ]
    b = [
        sum(y * x ** k for x, y in zip(x_values, y_values))
        for k in range(n + 1)
    ]

    return A, b


def least_squares(
        x_values: Vector,
        y_values: Vector,
        n: int
) -> Vector:
    A, b = normal_system(x_values, y_values, n)
    L, U, P, _ = LU_decomposition(A)
    a = gauss(P, L, U, b)

    return a


def F_n(
        x: float,
        a: Vector
) -> float:
    result = 0
    for i in range(len(a) - 1, -1, -1):
        result = result * x + a[i]

    return result


def Phi(
        x_values: Vector,
        y_values: Vector,
        a: Vector
) -> float:
    return sum((F_n(x, a) - y) ** 2 for x, y in zip(x_values, y_values))


def MSE(
        x_values: Vector,
        y_values: Vector,
        a: Vector
) -> float:

    return Phi(x_values, y_values, a) / len(x_values)


def main() -> None:
    x_values = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [-0.5, 0.0, 0.5, 0.86603, 1.0, 0.86603]

    errors = []
    for n in (1, 2, 3):
        a = least_squares(x_values, y_values, n)

        print(f"\nDegree: {n}")
        print("a:", [round(value, 9) for value in a])
        print(f"P_{n}(x) = " + " + ".join(f"{round(ai, 9)} * x^{i}" for i, ai in enumerate(a)))
        print(f"Phi = {round(Phi(x_values, y_values, a), 9)}")
        error = MSE(x_values, y_values, a)
        errors.append(error)

    best = errors.index(min(errors)) + 1
    print("\nComparison of MSE over all input points:")
    for n, error in enumerate(errors, start=1):
        print(f"P_{n}: MSE{n} = {format(error, ".9f").rstrip("0").rstrip(".")}")
    print(f"Best approximation by MSE: P_{best}(x), degree {best}")


if __name__ == '__main__':
    main()
