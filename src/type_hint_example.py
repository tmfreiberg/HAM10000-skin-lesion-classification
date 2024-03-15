import numpy as np
import typing


def data_fn(x: np.array) -> np.array:
    return np.array([1, 2, 3])


def get_transformer_fn(x: int) -> typing.Callable:
    def transformer_fn(y: int) -> int:
        return y + x
    return transformer_fn

