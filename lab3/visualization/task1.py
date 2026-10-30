from math import pi, sin

import matplotlib.pyplot as plt
import numpy as np

from lab3.task1 import lagrange_polynomial, newton_coefficients, newton_polynomial


def main() -> None:
    x_values_a = [0.1 * pi, 0.2 * pi, 0.3 * pi, 0.4 * pi]
    x_values_b = [0.1 * pi, pi / 6, 0.3 * pi, 0.4 * pi]
    x_test = pi / 4
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (name, x_values) in zip(axes, (("A", x_values_a), ("B", x_values_b))):
        y_values = [sin(x) for x in x_values]
        coefficients = newton_coefficients(x_values, y_values)
        grid = np.linspace(x_values[0], x_values[-1], 400)
        L_n = [lagrange_polynomial(x, x_values, y_values) for x in grid]
        P_n = [newton_polynomial(x, x_values, coefficients) for x in grid]

        ax.plot(grid, np.sin(grid), label='sin(x)')
        ax.plot(grid, L_n, '--', label='L_3(x)')
        ax.plot(grid, P_n, ':', label='P_3(x)')
        ax.scatter(x_values, y_values, label='Nodes', zorder=4)
        ax.scatter([x_test], [sin(x_test)], marker='x', label='x*', zorder=5)
        ax.set_title(f'Nodes {name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle('3.1. Lagrange and Newton interpolation')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
