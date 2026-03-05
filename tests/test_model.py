import pytest
from src.core.matrix import Matrix
from src.models.linear_regression import LinearRegression

X = Matrix([[1.0], [2.0], [3.0]])
Y = Matrix([[7.0], [9.0], [11.0]])

@pytest.fixture
def trained_model():
    model = LinearRegression(n_features=1)
    model.fit(X, Y, learning_rate=0.01, epochs=1500)
    return model

def test_model_fit_and_predict(trained_model):
    assert trained_model.weights[0, 0] == pytest.approx(2.0, abs=0.1)
    assert trained_model.bias == pytest.approx(5.0, abs=0.1)

    test_x = Matrix([[4.0]])
    prediction = trained_model.predict(test_x)

    assert prediction[0, 0] == pytest.approx(13.0, abs=0.1)

