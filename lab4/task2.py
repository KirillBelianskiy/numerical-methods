from math import exp

from lab1.task2 import sweep
from lab4.common import STEP, grid, print_errors, rk4_step


def f(
        x: float,
        y: float,
        z: float
) -> float:
    return y - 2 * z / x


def exact_solution(
        x: float
) -> float:
    return exp(x) / x


def initial_value(
        a: float,
        b: float,
        h: float,
        y0: float
) -> tuple[list[float], list[float], list[float]]:

    x_values = grid(a, b, h)
    y_values, z_values = [y0], [0.0]
    for x in x_values[:-1]:
        y, z = rk4_step(f, x, y_values[-1], z_values[-1], h)
        y_values.append(y)
        z_values.append(z)

    return x_values, y_values, z_values


def shooting(
        a: float,
        b: float,
        h: float
) -> tuple[list[float], list[float], list[float]]:
    _, y_values, z_values = initial_value(a, b, h, 1.0)
    residual_zero = -exp(2)
    residual_one = 1.5 * y_values[-1] + z_values[-1] - exp(2)
    y0 = 1 - residual_one / (residual_one - residual_zero)

    return initial_value(a, b, h, y0)


def finite_difference(
        a: float,
        b: float,
        h: float
) -> tuple[list[float], list[float]]:
    x_values = grid(a, b, h)
    n = len(x_values) - 1

    lower = [0.0] * n
    diagonal = [0.0] * (n + 1)
    upper = [0.0] * n
    right = [0.0] * (n + 1)
    for i in range(1, n):
        lower[i - 1] = 1 - h / x_values[i]
        diagonal[i] = -2 - h ** 2
        upper[i] = 1 + h / x_values[i]

    # Second-order boundary derivatives:
    # -3*y[0] + 4*y[1] - y[2] = 0;
    # y[n-2] - 4*y[n-1] + (3+3*h)*y[n] = 2*h*exp(2).
    # Eliminate y[2] and y[n-2] using the adjacent interior equations
    # to retain a tridiagonal system for the sweep method from lab1.
    diagonal[0] = -3 + lower[0] / upper[1]
    upper[0] = 4 + diagonal[1] / upper[1]
    lower[-1] = -4 - diagonal[n - 1] / lower[n - 2]
    diagonal[-1] = 3 + 3 * h - upper[n - 1] / lower[n - 2]
    right[-1] = 2 * h * exp(2)

    _, _, y_values = sweep(lower, diagonal, upper, right)
    return x_values, y_values


def main() -> None:
    a, b = 1.0, 2.0
    h = STEP

    print("4.2. Variant 1: x*y'' + 2*y' - x*y = 0")
    print("y'(1) = 0, 1.5*y(2) + y'(2) = exp(2), x in [1, 2]")
    print("Exact solution: exp(x)/x")
    print(f"h = {h}, h/2 = {h / 2}")
    x_values, y_values, z_values = shooting(a, b, h)
    _, fine_y, fine_z = shooting(a, b, h / 2)
    print_errors("Shooting (RK4 + secant method)", x_values, y_values, fine_y, exact_solution, 4)
    for step, y, z in ((h, y_values, z_values), (h / 2, fine_y, fine_z)):
        print(f"Boundary residuals, h = {step}: left = {abs(z[0]):.6e}, "
              f"right = {abs(1.5 * y[-1] + z[-1] - exp(2)):.6e}")

    x_values, y_values = finite_difference(a, b, h)
    _, fine_y = finite_difference(a, b, h / 2)
    print_errors("Finite differences", x_values, y_values, fine_y, exact_solution, 2)
    for step, y in ((h, y_values), (h / 2, fine_y)):
        left = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * step)
        right = 1.5 * y[-1] + (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * step) - exp(2)
        print(f"Boundary residuals, h = {step}: left = {abs(left):.6e}, right = {abs(right):.6e}")


if __name__ == '__main__':
    main()
