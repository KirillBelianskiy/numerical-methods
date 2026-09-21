from matrix_processing import Matrix, Vector

def multiply(
        A: Matrix,
        B: Matrix
) -> list:
    if not A or not B:
        raise ValueError("A and B must not be empty")

    if not isinstance(A[0], list):
        raise TypeError("First element of A is not a list")

    shape_A = len(A), len(A[0])

    if not isinstance(B[0], list):
        if shape_A[1] != len(B):
            raise ValueError('The dimensions are incompatible:\n',
                             f'A = {shape_A[0]}x{shape_A[1]}\nB = {len(B)}')

        C = [0] * shape_A[0]

        for i in range(shape_A[0]):
            for j in range(shape_A[1]):
                C[i] += A[i][j] * B[j]

        return C

    shape_B = len(B), len(B[0])

    if shape_A[1] != shape_B[0]:
        raise ValueError('The dimensions are incompatible:\n',
                         f'A = {shape_A[0]}x{shape_A[1]}\nB = {shape_B[0]}x{shape_B[1]}')

    C = [[0] * shape_B[1] for _ in range(shape_A[0])]

    for i in range(shape_A[0]):
        for j in range(shape_B[1]):
            for k in range(shape_B[0]):
                C[i][j] += A[i][k] * B[k][j]

    return C


def T(
        A: Matrix
) -> Matrix:
    n = len(A)
    m = len(A[0])
    new_A = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            new_A[j][i] = A[i][j]

    return new_A


def eye(
        size: int
) -> Matrix:
    matrix = [[0.0] * size for _ in range(size)]

    for i in range(size):
        matrix[i][i] = 1.0

    return matrix


def is_matrices_equal(
        A: Matrix,
        B: Matrix,
        eps: float = 1e-9
) -> bool:
    if len(A) != len(B):
        return False

    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False

        for j in range(len(A[i])):
            if abs(A[i][j] - B[i][j]) > eps:
                return False

    return True


def read_dense_matrix(
        filename: str
) -> Matrix:
    with open(filename, 'r') as f:
        m, n = map(int, f.readline().split())
        matrix = []
        for i in range(m):
            matrix.append(list(map(float, f.readline().split())))

        if m != len(matrix):
            raise ValueError("Matrix has wrong count of strings")

        for i in range(m):
            c_elem = len(matrix[i])
            if n != c_elem:
                raise ValueError("Matrix has wrong count of columns")

        return matrix


def pmprint(
        matrix: Matrix
) -> str:
    lines = ['[']

    for row in matrix:
        formatted_row = ' '.join(f'{x:.3f}' for x in row)
        lines.append(f'\t[{formatted_row}]')

    lines.append(']')

    return '\n'.join(lines)


def read_sparse_matrix(
        filename: str
) -> tuple[Vector, Vector, Vector, Vector]:
    with open(filename, 'r') as f:
        n = int(f.readline())

        a = list(map(float, f.readline().split()))
        b = list(map(float, f.readline().split()))
        c = list(map(float, f.readline().split()))
        d = list(map(float, f.readline().split()))

        if len(a) != n - 1 or len(b) != n or len(c) != n - 1 or len(d) != n:
            raise ValueError("Matrix has wrong count of elements")

        return a, b, c, d
