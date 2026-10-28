import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab1.task2 import sweep


def cubic_spline(
        x_values: list[float],
        y_values: list[float]
) -> list[tuple[float, float, float, float]]:
    n = len(x_values) - 1
    h = [x_values[i + 1] - x_values[i] for i in range(n)]

    # c[i] соответствует c_(i+1) учебника; c[0] = c[n] = 0.
    c = [0.0] * (n + 1)
    if n > 1:
        lower = h[1:-1]
        diagonal = [2 * (h[i - 1] + h[i]) for i in range(1, n)]
        upper = h[1:-1]
        right = [
            3 * (
                (y_values[i + 1] - y_values[i]) / h[i]
                - (y_values[i] - y_values[i - 1]) / h[i - 1]
            )
            for i in range(1, n)
        ]

        if n == 2:
            c[1] = right[0] / diagonal[0]
        else:
            _, _, c[1:-1] = sweep(lower, diagonal, upper, right)

    coefficients = []
    for i in range(n):
        a = y_values[i]
        b = (y_values[i + 1] - y_values[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * h[i])
        coefficients.append((a, b, c[i], d))

    return coefficients


def spline_interval(
        x: float,
        x_values: list[float]
) -> int:
    for i in range(len(x_values) - 1):
        if x_values[i] <= x <= x_values[i + 1]:
            return i

    raise ValueError("Spline interval was not found")


def spline_value(
        x: float,
        x_values: list[float],
        coefficients: list[tuple[float, float, float, float]]
) -> float:
    i = spline_interval(x, x_values)
    a, b, c, d = coefficients[i]
    dx = x - x_values[i]

    return a + b * dx + c * dx ** 2 + d * dx ** 3


def spline_second_derivative(
        x: float,
        x_values: list[float],
        coefficients: list[tuple[float, float, float, float]]
) -> float:
    i = spline_interval(x, x_values)
    _, _, c, d = coefficients[i]

    return 2 * c + 6 * d * (x - x_values[i])


def main() -> None:
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [0.0, 0.5, 0.86603, 1.0, 0.86603]
    x_test = 1.5

    coefficients = cubic_spline(x_values, y_values)
    S = spline_value(x_test, x_values, coefficients)

    print("Initial nodes:")
    print(" i          x_i             y_i")
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        print(f"{i:2d}  {x:12.8f}  {y:14.10f}")

    print(" i       [x_(i-1), x_i]              a_i             b_i             c_i             d_i")
    for i, (a, b, c, d) in enumerate(coefficients):
        print(
            f"{i + 1:2d}  [{x_values[i]:7.3f}, {x_values[i + 1]:7.3f}]"
            f"  {a:14.10f}  {b:14.10f}  {c:14.10f}  {d:14.10f}"
        )

    print("S_i(x) = a_i + b_i(x - x_(i-1)) + c_i(x - x_(i-1))^2 + d_i(x - x_(i-1))^3")
    print(f"S({x_test}) = {S:.10f}")
    i = spline_interval(x_test, x_values)
    print(f"Interval containing x*: {i + 1}, [{x_values[i]}, {x_values[i + 1]}]")

    eps = 1e-9
    print("\nInterpolation check:")
    print(" x_i            y_i          S(x_i)             error       passed")
    for x, y in zip(x_values, y_values):
        value = spline_value(x, x_values, coefficients)
        error = abs(value - y)
        print(f"{x:5.2f}  {y:14.10f}  {value:14.10f}  {error:14.6e}  {error <= eps}")

    left = spline_second_derivative(x_values[0], x_values, coefficients)
    right = spline_second_derivative(x_values[-1], x_values, coefficients)
    print(f"S''(x_0) = {left:.12e}, passed: {abs(left) <= eps}")
    print(f"S''(x_n) = {right:.12e}, passed: {abs(right) <= eps}")


if __name__ == '__main__':
    main()
