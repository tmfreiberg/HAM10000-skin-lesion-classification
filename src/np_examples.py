import numpy as np


if __name__ == '__main__':

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6, 7, 8, 9, 10])

    print(x[:, None]*y[None, :])

    assert np.all((x[:, None]*y[None,:]).shape == (3, 7))
