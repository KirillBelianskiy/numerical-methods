from math import ceil, factorial, floor, pi, prod, sin


def l(
        x: float,
        i: int,
        x_values: list[float]
) -> float:
    result = 1

    for j in range(len(x_values)):
        if i == j:
            continue

        result *= (x - x_values[j]) / (x_values[i] - x_values[j])

    return result


def lagrange_polynomial(
        x: float,
        x_values: list[float],
        y_values: list[float]
) -> float:
    result = 0

    for i in range(len(x_values)):
        result += y_values[i] * l(x, i, x_values)

    return result


def newton_coefficients(
        x_values: list[float],
        y_values: list[float]
) -> list[float]:

    differences = y_values.copy()
    coefficients = [differences[0]]

    for order in range(1, len(x_values)):
        for i in range(len(x_values) - order):
            differences[i] = (differences[i + 1] - differences[i]) / (
                x_values[i + order] - x_values[i]
            )

        coefficients.append(differences[0])

    return coefficients


def newton_polynomial(
        x: float,
        x_values: list[float],
        coefficients: list[float]
) -> float:
    result = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        result = coefficients[i] + (x - x_values[i]) * result

    return result


def df_4(
        x: float
) -> float:
    return sin(x)


def M(
        x_values: list[float]
) -> float:
    a, b = min(x_values), max(x_values)
    if ceil((a - pi / 2) / pi) <= floor((b - pi / 2) / pi):
        return 1.0

    return max(abs(df_4(a)), abs(df_4(b)))


def omega(
        x: float,
        x_values: list[float]
) -> float:
    result = 1

    for xi in x_values:
        result *= x - xi

    return result


def upper_estimate(
        x: float,
        x_values: list[float]
) -> float:
    return M(x_values + [x]) / factorial(4) * abs(omega(x, x_values))


def main() -> None:
    x_values_a = [0.1 * pi, 0.2 * pi, 0.3 * pi, 0.4 * pi]
    x_values_b = [0.1 * pi, pi / 6, 0.3 * pi, 0.4 * pi]
    x_test = pi / 4
    actual = sin(x_test)

    for name, x_values in (("A", x_values_a), ("B", x_values_b)):
        y_values = [sin(x) for x in x_values]
        count = len(x_values)

        table = [[None] * count for _ in range(count)]
        for i, y in enumerate(y_values):
            table[i][0] = y

        for order in range(1, count):
            for i in range(count - order):
                table[i][order] = (table[i + 1][order - 1] - table[i][order - 1]) / (
                    x_values[i + order] - x_values[i]
                )

        coefficients = newton_coefficients(x_values, y_values)
        L_n = lagrange_polynomial(x_test, x_values, y_values)
        P_n = newton_polynomial(x_test, x_values, coefficients)

        print(f"\nNodes {name}")
        print("Initial nodes and values:")
        for i, (xi, yi) in enumerate(zip(x_values, y_values)):
            print(f"  x_{i} = {round(xi, 9)}, y_{i} = {round(yi, 9)}")

        lagrange_terms = []
        for i, yi in enumerate(y_values):
            denominator = prod(
                x_values[i] - x_values[j] for j in range(count) if j != i
            )
            factors = " * ".join(
                f"(x - {round(x_values[j], 9)})" for j in range(count) if j != i
            )
            lagrange_terms.append(f"{round(yi / denominator, 9)} * {factors}")

        print("\nLagrange polynomial:")
        print("  L(x) = " + " + ".join(lagrange_terms))

        print("\nDivided differences:")
        print("i x_i f[x_i] differences")
        for i in range(count):
            row = [round(value, 9) for value in table[i] if value is not None]
            print(i, round(x_values[i], 9), row)

        newton_terms = [f"{round(coefficients[0], 9)}"]
        for order in range(1, count):
            factors = " * ".join(f"(x - {round(xi, 9)})" for xi in x_values[:order])
            newton_terms.append(f"{round(coefficients[order], 9)} * {factors}")

        print("\nNewton polynomial:")
        print("  P(x) = " + " + ".join(newton_terms))

        print(f"\nx* = {round(x_test, 9)}:")
        print(f"  L(x*) = {round(L_n, 9)}")
        print(f"  P(x*) = {round(P_n, 9)}")
        print(f"  f(x*) = {round(actual, 9)}")
        print(f"  |f(x*) - L(x*)| = {round(abs(actual - L_n), 9)}")
        print(f"  |f(x*) - P(x*)| = {round(abs(actual - P_n), 9)}")
        print(f"  |L(x*) - P(x*)| = {round(abs(L_n - P_n), 9)}")
        print(f"  Error bound = {round(upper_estimate(x_test, x_values), 9)}")


if __name__ == "__main__":
    main()
