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

NON_SQUARE_MATRIX = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]

SYMMETRIC_MATRIX = [
    [1.0, 2.0],
    [2.0, 1.0],
]


@pytest.fixture
def make_matrix():
    def _make(data):
        return Matrix(data).copy()

    return _make


def test_zeros_matrix():
    assert Matrix.zeros(2, 2) == Matrix([[0.0, 0.0], [0.0, 0.0]])


def test_ones_matrix():
    assert Matrix.ones(2, 2) == Matrix([[1.0, 1.0], [1.0, 1.0]])


def test_identity_matrix():
    assert Matrix.identity(2) == Matrix([[1.0, 0.0], [0.0, 1.0]])


def test_str(make_matrix):
    expected = "|  2.0  7.0  |\n|  8.0  3.0  |"
    assert str(make_matrix(MATRIX_DATA)) == expected


def test_repr(make_matrix):
    expected = "Matrix([[2.0, 7.0], [8.0, 3.0]])"
    assert repr(make_matrix(MATRIX_DATA)) == expected


def test_get_item(make_matrix):
    assert make_matrix(MATRIX_DATA)[1, 1] == 3.0


def test_get_item_raises_error(make_matrix):
    with pytest.raises(IndexError, match="Negative index"):
        make_matrix(MATRIX_DATA)[-1, -1]
    with pytest.raises(IndexError, match="Index out of bounds"):
        make_matrix(MATRIX_DATA)[4, 4]


def test_set_item(make_matrix):
    m = make_matrix(MATRIX_DATA)
    m[1, 1] = 6.0
    assert m[1, 1] == 6.0


def test_set_item_raises_error(make_matrix):
    with pytest.raises(IndexError, match="Negative index"):
        make_matrix(MATRIX_DATA)[-1, -1] = 1.0
    with pytest.raises(IndexError, match="Index out of bounds"):
        make_matrix(MATRIX_DATA)[4, 4] = 1.0


def test_add_scalar(make_matrix):
    assert (make_matrix(MATRIX_DATA) + 1).data == [[3.0, 8.0], [9.0, 4.0]]


def test_add_matrix(make_matrix):
    assert (make_matrix(MATRIX_DATA) + make_matrix(OTHER_MATRIX_DATA)).data == [
        [9.0, 14.0],
        [9.0, 5.0],
    ]


def test_add_matrix_raises_error(make_matrix):
    with pytest.raises(ValueError, match="Different dimensions"):
        make_matrix(MATRIX_DATA) + make_matrix(THIRD_ORDER_MATRIX)


def test_sub_scalar(make_matrix):
    assert (make_matrix(MATRIX_DATA) - 1).data == [[1.0, 6.0], [7.0, 2.0]]


def test_sub_matrix(make_matrix):
    assert (make_matrix(MATRIX_DATA) - make_matrix(OTHER_MATRIX_DATA)).data == [
        [-5.0, 0.0],
        [7.0, 1.0],
    ]


def test_sub_matrix_raises_error(make_matrix):
    with pytest.raises(ValueError, match="Different dimensions"):
        make_matrix(MATRIX_DATA) - make_matrix(THIRD_ORDER_MATRIX)


def test_mul_and_rmul_scalar(make_matrix):
    m = make_matrix(MATRIX_DATA)
    assert (m * 2).data == [[4.0, 14.0], [16.0, 6.0]]
    assert (2 * m).data == [[4.0, 14.0], [16.0, 6.0]]


def test_mul_and_rmul_matrix(make_matrix):
    m = make_matrix(MATRIX_DATA)
    other = make_matrix(OTHER_MATRIX_DATA)
    assert (m * other).data == [[21.0, 28.0], [59.0, 62.0]]
    assert (other * m).data == [[70.0, 70.0], [18.0, 13.0]]


def test_mul_and_rmul_raises_error(make_matrix):
    with pytest.raises(ValueError, match="Impossible multiplication"):
        make_matrix(MATRIX_DATA) * make_matrix(THIRD_ORDER_MATRIX)


def test_truediv(make_matrix):
    assert (make_matrix(MATRIX_DATA) / 2).data == [[1.0, 3.5], [4.0, 1.5]]


def test_truediv_raises_error(make_matrix):
    with pytest.raises(ValueError, match="Division by zero"):
        make_matrix(MATRIX_DATA) / 0


def test_neg(make_matrix):
    assert (-make_matrix(MATRIX_DATA)).data == [[-2.0, -7.0], [-8.0, -3.0]]


def test_pow(make_matrix):
    assert (make_matrix(MATRIX_DATA) ** 2).data == [[4.0, 49.0], [64.0, 9.0]]


def test_eq(make_matrix):
    m = make_matrix(MATRIX_DATA)
    assert m == m
    assert m != 1
    assert m != make_matrix(THIRD_ORDER_MATRIX)


def test_matrix_data_getter(make_matrix):
    assert make_matrix(MATRIX_DATA).data == MATRIX_DATA


def test_matrix_data_setter(make_matrix):
    m = make_matrix(MATRIX_DATA)
    m.data = OTHER_MATRIX_DATA
    assert m.data == OTHER_MATRIX_DATA


def test_matrix_data_setter_raises_errors(make_matrix):
    m = make_matrix(MATRIX_DATA)
    with pytest.raises(ValueError, match="Empty data was given"):
        m.data = []
    with pytest.raises(ValueError, match="Rows are not equal"):
        m.data = JAGGED_MATRIX


def test_shape_getter(make_matrix):
    assert make_matrix(MATRIX_DATA).shape == (2, 2)


def test_is_square(make_matrix):
    assert make_matrix(MATRIX_DATA).is_square()
    assert not make_matrix(NON_SQUARE_MATRIX).is_square()


def test_is_symmetric(make_matrix):
    assert make_matrix(SYMMETRIC_MATRIX).is_symmetric()
    assert not make_matrix(MATRIX_DATA).is_symmetric()
    assert not make_matrix(NON_SQUARE_MATRIX).is_symmetric()


def test_transpose(make_matrix):
    assert make_matrix(MATRIX_DATA).transpose().data == TRANSPOSED_MATRIX_DATA


def test_trace(make_matrix):
    assert make_matrix(MATRIX_DATA).trace() == 5.0


def test_trace_raises_error(make_matrix):
    with pytest.raises(ValueError, match="Not square matrix"):
        make_matrix(NON_SQUARE_MATRIX).trace()


def test_submatrix(make_matrix):
    result = make_matrix(THIRD_ORDER_MATRIX).submatrix(0, 0)
    assert result.data == [[2.0, 2.0], [3.0, 1.0]]


def test_det(make_matrix):
    assert make_matrix(MATRIX_DATA).det() == -50.0
    assert make_matrix(THIRD_ORDER_MATRIX).det() == 61.0


def test_det_raises_error(make_matrix):
    with pytest.raises(ValueError, match="Not square matrix"):
        make_matrix(NON_SQUARE_MATRIX).det()


def test_copy(make_matrix):
    m = make_matrix(MATRIX_DATA)
    c = m.copy()
    c[0, 0] = 99.0
    assert m[0, 0] == 2.0


def test_total(make_matrix):
    assert make_matrix(MATRIX_DATA).total() == 20.0


def test_mean(make_matrix):
    assert make_matrix(MATRIX_DATA).mean() == 5.0


def test_row(make_matrix):
    assert make_matrix(MATRIX_DATA).row(0) == [2.0, 7.0]


def test_col(make_matrix):
    assert make_matrix(MATRIX_DATA).col(0) == [2.0, 8.0]
