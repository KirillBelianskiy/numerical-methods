from math import sqrt


def f1(
        x1: float,
        x2: float
) -> float:
    return (x1 ** 2 + 4) * x2 - 8


def f2(
        x1: float,
        x2: float
) -> float:
    return (x1 - 1) ** 2 + (x2 - 1) ** 2 - 4


def phi(
        x1: float,
        x2: float
) -> tuple[float, float]:
    return (
        1 + sqrt(4 - (x2 - 1) ** 2),
        8 / (x1 ** 2 + 4)
    )


def norm(
        first: tuple[float, float],
        second: tuple[float, float]
) -> float:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def simple_iteration(
        x0: tuple[float, float],
        q: float,
        eps: float,
        max_iterations: int = 10000
) -> list[tuple[float, float]]:

    history = [x0]
    for _ in range(max_iterations):

        x_new = phi(*history[-1])

        history.append(x_new)

        if q / (1 - q) * norm(x_new, history[-2]) <= eps:
            return history

    print("Simple iteration method did not converge")
    return history


def newton(
        x0: tuple[float, float],
        eps: float,
        max_iterations: int = 10000
) -> list[tuple[float, float]]:

    history = [x0]
    for _ in range(max_iterations):

        x1, x2 = history[-1]
        first, second = f1(x1, x2), f2(x1, x2)

        j11 = 2 * x1 * x2
        j12 = x1 ** 2 + 4
        j21 = 2 * (x1 - 1)
        j22 = 2 * (x2 - 1)

        determinant = j11 * j22 - j12 * j21

        dx1 = (-first * j22 + j12 * second) / determinant
        dx2 = (first * j21 - j11 * second) / determinant
        x_new = (x1 + dx1, x2 + dx2)

        history.append(x_new)

        if norm(x_new, history[-2]) < eps:
            return history

    print("Newton method did not converge")
    return history


def reference_solution() -> tuple[float, float]:
    def residual(
            x1: float
    ) -> float:
        x2 = 8 / (x1 ** 2 + 4)
        return f2(x1, x2)

    left, right = 2.8, 3.0
    for _ in range(60):
        middle = (left + right) / 2
        if residual(left) * residual(middle) <= 0:
            right = middle
        else:
            left = middle

    x1 = (left + right) / 2
    return x1, 8 / (x1 ** 2 + 4)


def print_errors(
        name: str,
        history: list[tuple[float, float]],
        reference: tuple[float, float]
) -> None:
    print(f"\n{name}: {history[-1]}, iterations: {len(history) - 1}")
    print(" k          x1^(k)          x2^(k)      error (max norm)       residual")
    for k, (x1, x2) in enumerate(history):
        error = norm((x1, x2), reference)
        residual = max(abs(f1(x1, x2)), abs(f2(x1, x2)))
        print(f"{k:2d}  {x1:15.12f}  {x2:15.12f}  {error:18.10e}  {residual:14.10e}")


def main() -> None:
    x0 = (2.9, 0.7)
    eps = 0.01
    q = max(0.4 / sqrt(4 - 0.4 ** 2), 16 * 2.8 / (2.8 ** 2 + 4) ** 2)

    reference = reference_solution()

    print(f"Initial approximation: {x0}")
    print(f"Contraction coefficient: q = {q}")
    print_errors("Simple iteration method", simple_iteration(x0, q, eps), reference)
    print_errors("Newton method", newton(x0, eps), reference)


if __name__ == "__main__":
    main()
