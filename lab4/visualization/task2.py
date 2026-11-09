from argparse import ArgumentParser
from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np

from lab4.common import STEP
from lab4.task2 import exact_solution, finite_difference, shooting


def main() -> None:
    parser = ArgumentParser(description="Plot solutions for task 4.2")
    parser.add_argument('--no-show', action='store_true', help="Save graph without opening a window")
    args = parser.parse_args()

    a, b = 1.0, 2.0
    h = STEP
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, method in (
            ('shooting method', shooting),
            ('finite difference method', finite_difference)
    ):
        for step in (h, h / 2):
            x, y, *_ = method(a, b, step)
            ax.scatter(x, y, label=f'{name}, step={step:g}')

    x = np.linspace(a, b, 500)
    ax.plot(x, [exact_solution(t) for t in x], color='tab:blue', label='exact solution')
    ax.legend(loc='upper left')

    output = join(dirname(abspath(__file__)), 'task2_graph.png')
    fig.savefig(output, dpi=180)
    print(f'Graph saved: {output}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
