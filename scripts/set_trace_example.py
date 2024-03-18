import pathlib
import os


if __name__ == '__main__':

    path = pathlib.Path("/mnt/data/HAM1000")

    image_path = path.joinpath("images")

    import ipdb; ipdb.set_trace()

    print("image_path = ", image_path)
