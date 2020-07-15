import cv2
from logging import getLogger, INFO
from logzero import logger
import copy
import numpy as np

higher_res = cv2.imread('img/sample.jpg')

d = 3
pylamid = []

lower_img = copy.deepcopy(higher_res)
# lower_img = float(lower_img)/255
# lower_img = np.exp(lower_img)
for i in range(d+1):
    lower_img = cv2.pyrDown(lower_img)
    lower_img_ap = np.pad(lower_img, ((1, 1), (1, 1), (0, 0)), "edge")
    dy = (lower_img_ap[1:, :] - lower_img_ap[:-1, :])/2**(i+1)
    dx = (lower_img_ap[:, 1:] - lower_img_ap[:, :-1])/2**(i+1)

    pylamid.append([dx, dy])
    logger.info(f"{lower_img.shape=}")
    cv2.imwrite(f"img/{d=}_test.jpg", lower_img)

logger.info(f"{higher_res.shape}")

# cv2.imwrite("img/test.jpg", lower_img)
