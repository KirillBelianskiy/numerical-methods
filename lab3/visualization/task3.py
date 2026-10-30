import matplotlib.pyplot as plt
import numpy as np

from lab3.task3 import MSE, F_n, least_squares


def main() -> None:
    x_values = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [-0.5, 0.0, 0.5, 0.86603, 1.0, 0.86603]
    grid = np.linspace(x_values[0], x_values[-1], 400)

    plt.figure(figsize=(9, 5))
    plt.scatter(x_values, y_values, label='Input data', zorder=4)
    for n in (1, 2, 3):
        a = least_squares(x_values, y_values, n)
        values = [F_n(x, a) for x in grid]
        plt.plot(grid, values, label=f'P_{n}(x), MSE{n} = {format(MSE(x_values, y_values, a), ".9f").rstrip("0").rstrip(".")}')

    plt.title('3.3. Least squares')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
