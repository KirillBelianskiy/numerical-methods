from math import cos, sin

from lab4.common import STEP, grid, print_errors, rk4_step


def f(
        x: float,
        y: float,
        z: float
) -> float:
    return sin(3 * x) - y


def exact_solution(
        x: float
) -> float:
    return cos(x) + 11 / 8 * sin(x) - sin(3 * x) / 8


def euler(
        a: float,
        b: float,
        h: float,
        y0: float,
        z0: float
) -> tuple[list[float], list[float], list[float]]:

    x_values = grid(a, b, h)
    y_values, z_values = [y0], [z0]
    for x in x_values[:-1]:
        y, z = y_values[-1], z_values[-1]
        y_values.append(y + h * z)
        z_values.append(z + h * f(x, y, z))

    return x_values, y_values, z_values


def runge_kutta(
        a: float,
        b: float,
        h: float,
        y0: float,
        z0: float
) -> tuple[list[float], list[float], list[float]]:

    x_values = grid(a, b, h)
    y_values, z_values = [y0], [z0]
    for x in x_values[:-1]:
        y, z = rk4_step(f, x, y_values[-1], z_values[-1], h)
        y_values.append(y)
        z_values.append(z)

    return x_values, y_values, z_values


def adams(
        a: float,
        b: float,
        h: float,
        y0: float,
        z0: float
) -> tuple[list[float], list[float], list[float]]:

    x_values = grid(a, b, h)
    y_values, z_values = [y0], [z0]
    for x in x_values[:3]:
        y, z = rk4_step(f, x, y_values[-1], z_values[-1], h)
        y_values.append(y)
        z_values.append(z)

    coefficients = [55, -59, 37, -9]
    for i in range(3, len(x_values) - 1):
        y = y_values[i] + h / 24 * sum(
            c * z_values[i - j] for j, c in enumerate(coefficients)
        )
        z = z_values[i] + h / 24 * sum(
            c * f(x_values[i - j], y_values[i - j], z_values[i - j])
            for j, c in enumerate(coefficients)
        )
        y_values.append(y)
        z_values.append(z)

    return x_values, y_values, z_values


def main() -> None:
    a, b = 0.0, 1.0
    y0, z0 = 1.0, 1.0
    h = STEP

    print("4.1. Variant 1: y'' + y - sin(3x) = 0")
    print("y(0) = 1, y'(0) = 1, x in [0, 1]")
    print("Exact solution: cos(x) + 11/8*sin(x) - sin(3*x)/8")
    print(f"h = {h}, h/2 = {h / 2}")
    for name, method, p in (
            ("Euler", euler, 1),
            ("Runge-Kutta 4", runge_kutta, 4),
            ("Adams 4", adams, 4)
    ):
        x_values, y_values, _ = method(a, b, h, y0, z0)
        _, fine_y, _ = method(a, b, h / 2, y0, z0)
        print_errors(name, x_values, y_values, fine_y, exact_solution, p)


if __name__ == '__main__':
    main()
