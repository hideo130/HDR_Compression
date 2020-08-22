import numpy as np
from logzero import logger
from pathlib import Path
import sys
import_path = Path(__file__).resolve().parents[1]
sys.path.append(str(import_path))
from hdr_comp import get_coeff, get_b, get_DLU
# print(sys.path)


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


def check_get_coeff():
    # coeff_x, coeff_y = get_coeff(3024-2, 4032-2)
    coeff_x, coeff_y = get_coeff(766, 510)


def check_get_DLU():
    D, L, U = get_DLU(4, 3)
    logger.info(D.toarray())
    logger.info(L.toarray())
    logger.info(U.toarray())


if __name__ == "__main__":
    # print_coeff()
    # print_b()
    # check_get_coeff()
    check_get_DLU()