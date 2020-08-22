from hdr_comp import get_coeff, get_b
import numpy as np
from logzero import logger
from pathlib import Path
import sys
import_path = Path(__file__).resolve().parents[1]
sys.path.append(str(import_path))


def print_coeff():
    coeff_x, coeff_y = get_coeff(3, 3)
    logger.info(f"{coeff_x=}")
    logger.info(f"{coeff_y=}")
    logger.info(f"{coeff_x.shape=}")
    logger.info(f"{coeff_y.shape=}")


def print_b():
    img = np.arange(25).reshape((5, 5))
    img[1:-1, 1:-1] = 0
    tmp = np.zeros((3, 3))
    get_b(img, tmp, tmp)


if __name__ == "__main__":
    # print_coeff()
    print_b()
