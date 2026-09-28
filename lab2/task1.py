from math import isfinite, log, sqrt


def f(
        x: float
) -> float:
    return 2 ** x - x ** 2 - 0.5


def df(
        x: float
) -> float:
    return 2 ** x * log(2) - 2 * x


def d2f(
        x: float
) -> float:
    return 2 ** x * log(2) ** 2 - 2


def phi(
        x: float
) -> float:
    return sqrt(2 ** x - 0.5)


def simple_iteration(
        x0: float,
        q: float,
        eps: float,
        max_iterations: int = 10000
) -> list[float]:

    history = [x0]
    for _ in range(max_iterations):
        x_new = phi(history[-1])
        if not isfinite(x_new):
            print("Simple iteration method diverged")
            return history
        history.append(x_new)

        if q / (1 - q) * abs(x_new - history[-2]) <= eps:
            return history

    print("Simple iteration method did not converge within the specified number of steps.")
    return history


def choose_newton_x0(
        a: float,
        b: float
) -> float:
    for x in (a, b):
        if f(x) * d2f(x) > 0:
            return x

    print("Condition of Newton’s method is not met at the endpoints of the segment.")
    return (a + b) / 2


def newton(
        x0: float,
        eps: float,
        max_iterations: int = 10000
) -> list[float]:
    history = [x0]
    for _ in range(max_iterations):

        x_new = history[-1] - f(history[-1]) / df(history[-1])
        history.append(x_new)

        if abs(x_new - history[-2]) < eps:
            return history

    return history


def reference_root(
        a: float,
        b: float
) -> float:
    for _ in range(60):
        middle = (a + b) / 2
        if f(a) * f(middle) <= 0:
            b = middle
        else:
            a = middle
    return (a + b) / 2


def print_errors(
        name: str,
        history: list[float],
        root: float
) -> None:
    print(f"\n{name}: {history[-1]:.15f}, iterations: {len(history) - 1}")
    print(" k             x^(k)       |x^(k) - x*|         |f(x^(k))|")
    for k, x in enumerate(history):
        print(f"{k:2d}  {x:18.15f}  {abs(x - root):18.10e}  {abs(f(x)):18.10e}")


def main() -> None:
    a, b = 1.5, 1.6
    eps = 0.01
    q = 2 ** b * log(2) / (2 * phi(b))
    x0_si = (a + b) / 2
    x0_newton = choose_newton_x0(a, b)
    print("x_0 for simple iteration method: ", x0_si)
    print("x_0 for newton method: ", x0_newton)
    print("phi(x)=sqrt(2**x - 0.5)", f"q={q}", sep="\n")

    root = reference_root(a, b)
    print_errors("Simple iterations method", simple_iteration(x0_si, q, eps), root)
    print_errors("Newton method", newton(x0_newton, eps), root)
    

if __name__ == "__main__":
    main()
