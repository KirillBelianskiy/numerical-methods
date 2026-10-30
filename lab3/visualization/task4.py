import matplotlib.pyplot as plt
import numpy as np

from lab3.task4 import derivatives, divided_differences


def main() -> None:
    x_values = [-1.0, 0.0, 1.0, 2.0, 3.0]
    y_values = [-0.5, 0.0, 0.5, 0.86603, 1.0]
    x_test = 1.0
    i = 1
    first, _, difference = divided_differences(x_values, y_values, i)
    df, d2f = derivatives(x_test, x_values, y_values, i)
    grid = np.linspace(x_values[i], x_values[i + 2], 400)
    phi = [
        y_values[i] + first * (x - x_values[i])
        + difference * (x - x_values[i]) * (x - x_values[i + 1])
        for x in grid
    ]
    tangent = [y_values[i + 1] + df * (x - x_test) for x in grid]

    plt.figure(figsize=(9, 5))
    plt.plot(x_values, y_values, 'o:', label='Tabulated function')
    plt.plot(grid, phi, label='phi(x): quadratic polynomial')
    plt.plot(grid, tangent, '--', label=f"Tangent, y'(1) = {round(df, 9)}")
    plt.axvline(x_test, linestyle=':', alpha=0.5)
    plt.title(f"3.4. Numerical differentiation, y''(1) = {round(d2f, 9)}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
