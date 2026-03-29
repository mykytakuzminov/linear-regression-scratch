import pytest

from linlag import Matrix

MATRIX_DATA = [
    [2.0, 7.0],
    [8.0, 3.0],
]

TRANSPOSED_MATRIX_DATA = [
    [2.0, 8.0],
    [7.0, 3.0],
]

OTHER_MATRIX_DATA = [
    [7.0, 7.0],
    [1.0, 2.0],
]

JAGGED_MATRIX = [
    [3.0, 1.0],
    [9.0, 2.0, 7.0],
    [2.0],
]

THIRD_ORDER_MATRIX = [
    [3.0, 8.0, 1.0],
    [5.0, 2.0, 2.0],
    [7.0, 3.0, 1.0],
]


@pytest.fixture
def populated_matrix():
    return Matrix([row[:] for row in MATRIX_DATA])


@pytest.fixture
def populated_other():
    return Matrix([row[:] for row in MATRIX_DATA])


def test_zeros_matrix():
    assert Matrix.zeros(2, 2) == Matrix([[0.0, 0.0], [0.0, 0.0]])


def test_ones_matrix():
    assert Matrix.ones(2, 2) == Matrix([[1.0, 1.0], [1.0, 1.0]])


def test_identity_matrix():
    assert Matrix.identity(2) == Matrix([[1.0, 0.0], [0.0, 1.0]])


def test_str(populated_matrix):
    expected = "|  2.0  7.0  |\n|  8.0  3.0  |"
    assert str(populated_matrix) == expected


def test_get_item(populated_matrix):
    assert populated_matrix[1, 1] == 3.0


def test_get_item_raises_error(populated_matrix):
    with pytest.raises(IndexError, match="Negative index"):
        populated_matrix[-1, -1]
    with pytest.raises(IndexError, match="Index out of bounds"):
        populated_matrix[4, 4]


def test_set_item(populated_matrix):
    populated_matrix[1, 1] = 6.0
    assert populated_matrix[1, 1] == 6.0


def test_set_item_raises_error(populated_matrix):
    with pytest.raises(IndexError, match="Negative index"):
        populated_matrix[-1, -1] = 1.0
    with pytest.raises(IndexError, match="Index out of bounds"):
        populated_matrix[4, 4] = 1.0


def test_add_scalar(populated_matrix):
    assert (populated_matrix + 1).data == [[3.0, 8.0], [9.0, 4.0]]


def test_add_matrix(populated_matrix):
    other = Matrix(OTHER_MATRIX_DATA)
    assert (populated_matrix + other).data == [[9.0, 14.0], [9.0, 5.0]]


def test_add_matrix_raises_error(populated_matrix):
    other = Matrix(THIRD_ORDER_MATRIX)
    with pytest.raises(ValueError, match="Different dimensions"):
        populated_matrix + other


def test_sub_scalar(populated_matrix):
    assert (populated_matrix - 1).data == [[1.0, 6.0], [7.0, 2.0]]


def test_sub_matrix(populated_matrix):
    other = Matrix(OTHER_MATRIX_DATA)
    assert (populated_matrix - other).data == [[-5.0, 0.0], [7.0, 1.0]]


def test_sub_matrix_raises_error(populated_matrix):
    other = Matrix(THIRD_ORDER_MATRIX)
    with pytest.raises(ValueError, match="Different dimensions"):
        populated_matrix - other


def test_mul_and_rmul_scalar(populated_matrix):
    assert (populated_matrix * 2).data == [[4.0, 14.0], [16.0, 6.0]]
    assert (2 * populated_matrix).data == [[4.0, 14.0], [16.0, 6.0]]


def test_mul_and_rmul_matrix(populated_matrix):
    other = Matrix(OTHER_MATRIX_DATA)
    assert (populated_matrix * other).data == [[21.0, 28.0], [59.0, 62.0]]
    assert (other * populated_matrix).data == [[70.0, 70.0], [18.0, 13.0]]


def test_mul_and_rmul_raises_error(populated_matrix):
    other = Matrix(THIRD_ORDER_MATRIX)
    with pytest.raises(ValueError, match="Impossible multiplication"):
        populated_matrix * other


def test_pow(populated_matrix):
    assert (populated_matrix**2).data == [[4.0, 49.0], [64.0, 9.0]]


def test_eq(populated_matrix):
    assert populated_matrix == populated_matrix
    assert populated_matrix != 1
    assert populated_matrix != Matrix(THIRD_ORDER_MATRIX)


def test_matrix_data_getter(populated_matrix):
    assert populated_matrix.data == MATRIX_DATA


def test_matrix_data_setter(populated_matrix):
    populated_matrix.data = OTHER_MATRIX_DATA
    assert populated_matrix.data == OTHER_MATRIX_DATA


def test_matrix_data_setter_raises_errors(populated_matrix):
    with pytest.raises(ValueError, match="Empty data was given"):
        populated_matrix.data = []
    with pytest.raises(ValueError, match="Rows are not equal"):
        populated_matrix.data = JAGGED_MATRIX


def test_shape_getter(populated_matrix):
    assert populated_matrix.shape == (2, 2)


def test_tanspose(populated_matrix):
    assert populated_matrix.transpose().data == TRANSPOSED_MATRIX_DATA


def test_sum(populated_matrix):
    matrix_sum = sum([sum(row) for row in MATRIX_DATA])
    assert populated_matrix.sum() == matrix_sum


def test_mean(populated_matrix):
    matrix_mean = sum([sum(row) for row in MATRIX_DATA]) / 4
    assert populated_matrix.mean() == matrix_mean
