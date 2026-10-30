import matplotlib.pyplot as plt
import numpy as np

from lab3.task2 import cubic_spline, spline_value


def main() -> None:
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [0.0, 0.5, 0.86603, 1.0, 0.86603]
    x_test = 1.5
    coefficients = cubic_spline(x_values, y_values)
    result = spline_value(x_test, x_values, coefficients)

    plt.figure(figsize=(9, 5))
    for i, (a, b, c, d) in enumerate(coefficients):
        grid = np.linspace(x_values[i], x_values[i + 1], 100)
        dx = grid - x_values[i]
        S = a + b * dx + c * dx ** 2 + d * dx ** 3
        plt.plot(grid, S, label=f'S_{i + 1}(x), [{round(x_values[i], 9)}, {round(x_values[i + 1], 9)}]')
    plt.scatter(x_values, y_values, label='Nodes', zorder=4)
    plt.scatter([x_test], [result], label=f'S(1.5) = {round(result, 9)}', zorder=5)
    plt.title('3.2. Natural cubic spline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
