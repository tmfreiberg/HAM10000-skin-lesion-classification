import numpy as np
from graph_utils.math import nmb_images


if __name__ == '__main__':

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6, 7, 8, 9, 10])

    print(x[:, None]*y[None, :])

    assert np.all((x[:, None]*y[None,:]).shape == (3, 7))

    print("I imported nmb_images from a sub-package. It equals = ", nmb_images)
