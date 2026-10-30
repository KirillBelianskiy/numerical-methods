def divided_differences(
        x_values: list[float],
        y_values: list[float],
        i: int
) -> tuple[float, float, float]:

    first = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
    second = (y_values[i + 2] - y_values[i + 1]) / (x_values[i + 2] - x_values[i + 1])
    difference = (second - first) / (x_values[i + 2] - x_values[i])

    return first, second, difference


def derivatives(
        x: float,
        x_values: list[float],
        y_values: list[float],
        i: int
) -> tuple[float, float]:
    first, _, difference = divided_differences(x_values, y_values, i)

    df = first + difference * (2 * x - x_values[i] - x_values[i + 1])
    d2f = 2 * difference

    return df, d2f


def main() -> None:
    x_values = [-1.0, 0.0, 1.0, 2.0, 3.0]
    y_values = [-0.5, 0.0, 0.5, 0.86603, 1.0]
    x_test = 1.0
    i = 1

    left, right, _ = divided_differences(x_values, y_values, i)
    df, d2f = derivatives(x_test, x_values, y_values, i)

    print("Initial table:")
    print(" i x_i y_i")
    for j, (x, y) in enumerate(zip(x_values, y_values)):
        print(f"{j}  {round(x, 9)}  {round(y, 9)}")
    print(f"x* = {x_test}")
    print("Selected node indices:", list(range(i, i + 3)))
    print("Selected values:", y_values[i:i + 3])
    print("Selected nodes:", x_values[i:i + 3])
    print(f"Left secant slope: {round(left, 9)}")
    print(f"Right secant slope: {round(right, 9)}")
    print(f"y'({x_test}) = {round(df, 9)}")
    print(f"y''({x_test}) = {round(d2f, 9)}")


if __name__ == '__main__':
    main()
