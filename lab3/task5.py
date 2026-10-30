from math import isclose, log


def f(
        x: float
) -> float:
    return x / (2 * x + 5)


def integration_nodes(
        a: float,
        b: float,
        h: float
) -> list[float]:

    n = round((b - a) / h)

    return [a + i * h for i in range(n)] + [b]


def left_rectangles(
        a: float,
        b: float,
        h: float
) -> float:
    x_values = integration_nodes(a, b, h)
    F = 0
    for x in x_values[:-1]:
        F += h * f(x)

    return F


def right_rectangles(
        a: float,
        b: float,
        h: float
) -> float:
    x_values = integration_nodes(a, b, h)
    F = 0
    for x in x_values[1:]:
        F += h * f(x)

    return F


def rectangles(
        a: float,
        b: float,
        h: float
) -> float:
    x_values = integration_nodes(a, b, h)
    F = 0
    for i in range(1, len(x_values)):
        F += h * f((x_values[i - 1] + x_values[i]) / 2)

    return F


def trapezoids(
        a: float,
        b: float,
        h: float
) -> float:
    x_values = integration_nodes(a, b, h)
    F = h * (f(a) + f(b)) / 2
    for x in x_values[1:-1]:
        F += h * f(x)

    return F


def simpson(
        a: float,
        b: float,
        h: float
) -> float:
    x_values = integration_nodes(a, b, h)
    n = len(x_values) - 1

    F = f(a) + f(b)
    for i in range(1, n):
        F += (4 if i % 2 else 2) * f(x_values[i])

    return h * F / 3


def runge_romberg(
        F_h: float,
        F_kh: float,
        k: float,
        p: int
) -> tuple[float, float]:

    correction = (F_h - F_kh) / (k ** p - 1)
    F = F_h + correction
    R = abs(correction)

    return F, R


def antiderivative(
        x: float
) -> float:
    return x / 2 - 5 / 4 * log(abs(2 * x + 5))


def main() -> None:
    a, b = -1.0, 1.0
    h1, h2 = 0.5, 0.25
    reference = antiderivative(b) - antiderivative(a)

    print(f"Interval: [{a}, {b}], h1 = {h1}, h2 = {h2}")
    print(f"Reference integral: {round(reference, 9)}")
    print(
        "Method F_h1 F_h2 p F_Runge"
        " R_h2 error_h1 error_h2 error_Runge"
    )
    results = []
    for name, method, p in (
            ("Left rectangles", left_rectangles, 1),
            ("Right rectangles", right_rectangles, 1),
            ("Middle rectangles", rectangles, 2),
            ("Trapezoids", trapezoids, 2),
            ("Simpson", simpson, 4)
    ):
        F_kh = method(a, b, h1)
        F_h = method(a, b, h2)
        F, R = runge_romberg(F_h, F_kh, h1 / h2, p)
        error_h1 = abs(F_kh - reference)
        error_h2 = abs(F_h - reference)
        error_Runge = abs(F - reference)
        results.append((name, error_h1, error_h2, error_Runge))
        print(
            f"{name}  {round(F_kh, 9)}  {round(F_h, 9)}  {p}  {round(F, 9)}"
            f"  {format(R, ".9f").rstrip("0").rstrip(".")}  {format(error_h1, ".9f").rstrip("0").rstrip(".")}  {format(error_h2, ".9f").rstrip("0").rstrip(".")}  {format(error_Runge, ".9f").rstrip("0").rstrip(".")}"
        )

    for column, label in ((1, "h1"), (2, "h2"), (3, "Runge-Romberg")):
        best = results[0]
        for result in results[1:]:
            if result[column] < best[column]:
                best = result
        print(f"Best method ({label}): {best[0]}, error = {format(best[column], ".9f").rstrip("0").rstrip(".")}")


if __name__ == '__main__':
    main()
