from src.core.matrix import Matrix


def mse(y_true: Matrix, y_pred: Matrix) -> float:
    ""
    loss = ((y_true - y_pred) ** 2).mean()
    return loss

