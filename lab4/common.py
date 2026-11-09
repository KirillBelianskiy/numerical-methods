from collections.abc import Callable

STEP = 0.1


def grid(
        a: float,
        b: float,
        h: float
) -> list[float]:
    n = round((b - a) / h)
    return [a + i * h for i in range(n)] + [b]


def rk4_step(
        f: Callable[[float, float, float], float],
        x: float,
        y: float,
        z: float,
        h: float
) -> tuple[float, float]:
    k1, l1 = h * z, h * f(x, y, z)
    k2 = h * (z + l1 / 2)
    l2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)
    k3 = h * (z + l2 / 2)
    l3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)
    k4 = h * (z + l3)
    l4 = h * f(x + h, y + k3, z + l3)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6, z + (l1 + 2 * l2 + 2 * l3 + l4) / 6


def print_errors(
        name: str,
        x_values: list[float],
        y_values: list[float],
        fine_values: list[float],
        exact: Callable[[float], float],
        p: int
) -> None:
    print(f"\n{name}, p = {p}")
    print("     x          y_h        y_h/2       y_exact      error_h    error_h/2      RR_h/2")
    estimates = []
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        fine = fine_values[2 * i]
        reference = exact(x)
        estimate = abs(fine - y) / (2 ** p - 1)
        estimates.append(estimate)
        print(
            f"{x:6.3f}  {y:11.8f}  {fine:11.8f}  {reference:11.8f}"
            f"  {abs(y - reference):11.3e}  {abs(fine - reference):11.3e}"
            f"  {estimate:11.3e}"
        )

    coarse_error = max(abs(y - exact(x)) for x, y in zip(x_values, y_values))
    fine_h = (x_values[-1] - x_values[0]) / (len(fine_values) - 1)
    fine_error = max(
        abs(y - exact(x_values[0] + i * fine_h))
        for i, y in enumerate(fine_values)
    )
    print(f"Maximum absolute error, h: {coarse_error:.6e}")
    print(f"Maximum absolute error, h/2: {fine_error:.6e}")
    print(f"Runge-Romberg estimate for h/2 at common nodes: {max(estimates):.6e}")
