import cv2
from logging import getLogger, INFO
from logzero import logger
import copy
import numpy as np
from scipy import signal
from tqdm import tqdm


def upsampling(gradient):
    """畳み込みを利用して線形補完でアップサンプリングする.


    Args:
        gradient (numpy型): shape:(w,h,_)

    Returns:
        upsamplingした勾配: shape:(2w,2h,_)
    """
    h, w, _ = gradient.shape
    new_gradient = np.zeros((2*h, 2*w, _))
    new_gradient[0::2, 0::2, :] = gradient[:, :, :]
    linear_filter = np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]])
    for i in range(3):
        new_gradient[..., i] = signal.convolve2d(
            new_gradient[..., i], linear_filter, boundary='symm', mode='same')
    return new_gradient


def cal_divG(Gx, Gy):
    """divGを計算する
        G=(Gx, Gy)
        入力画像サイズを(w,h,3)とする．
    Args:
        Gx (numpy): ベクトル場のx成分 shape=(w,h,3)
        Gy (numpy): ベクトル場のy成分 shape=(w,h,3)

    Returns:
        divG: dGx,dGyのshapeは(w-2, h-2,3)
    """
    padding_Gx = np.pad(Gx, ((0, 0), (1, 1), (0, 0)), "edge")
    dGx = (padding_Gx[:, 1:-1] - padding_Gx[:, :-2])
    # 端は使わない
    dGx = dGx[1:-1, 1:-1]
    padding_Gy = np.pad(Gy, ((1, 1), (0, 0), (0, 0)), "edge")
    dGy = (padding_Gy[1:-1, ...] - padding_Gy[:-2, ...])
    # 端は使わない
    dGy = dGy[1:-1, 1:-1]
    return dGx, dGy


def get_coeff(h, w):
    eye = np.eye(w*h)
    # x方向の微分作用素を求める
    upper = np.eye(w*h)
    upper[::w, ::w] = 0
    upper = np.vstack((upper[1:], np.array([0]*w*h)))
    coeff_x = upper + -2*eye + upper.T

    # y方向の微分作用素を求める
    zeros = np.zeros((w, w*h))
    upper = np.vstack((eye[w:, :], zeros))
    lower = upper.T
    coeff_y = upper + -2*eye + lower
    return coeff_x, coeff_y


def get_b(boundary, dGx, dGy):
    # 元の画像が(w, h)なら勾配を操作する範囲は(w-2, h-2)
    # boundaryのshapeは(w,h)だからx,y方向に二つ余分に0がある.

    # x微分のとき必要なのは左端と右端
    # 上と下の端は削除
    boundary_x = boundary[1:-1, :]  # (w,h)->(w, h-2)

    # (w, h-2) -> (w-2, h-2)
    boundary_x = np.delete(boundary_x, [1, 2], axis=1)
    boundary_x = boundary_x.reshape((-1, 1))

    # y微分のときに必要なのは上と下の端
    # 左右の端は削除
    boundary_y = boundary[:, 1:-1]  # (w,h)->(w-2, h)
    # (w-2, h) -> (w-2, h-2)
    boundary_y = np.delete(boundary_y, [1, 2], axis=0)
    boundary_y = boundary_y.reshape((-1, 1))

    logger.debug(f"{boundary_x=}")
    logger.debug(f"{boundary_x.shape=}")
    logger.debug(f"{boundary_y=}")
    logger.debug(f"{boundary_y.shape=}")

    return boundary_x + boundary_y + dGx.reshape((-1, 1)) + dGy.reshape((-1, 1))


def gauss_seidel(M, N, b, k_max, epsilon, x):
    success = False
    for i in tqdm(range(k_max)):
        new_x = np.dot(M, x) + np.dot(N, b)
        if np.linalg.norm(new_x-x) < epsilon:
            success = True
            break
        x = new_x

    if success:
        return new_x
    else:
        print("失敗")
        return new_x


if __name__ == "__main__":
    higres_img = cv2.imread('img/sample.jpg')
    logger.info(f"{higres_img.shape}")
    d = 4
    pylamid = []

    higres_img = np.float32(higres_img) / 255
    higres_img = np.exp(higres_img)

    lower_img = copy.deepcopy(higres_img)
    for i in range(d+1):
        lower_img_ap = np.pad(lower_img, ((1, 1), (1, 1), (0, 0)), "edge")
        dy = (lower_img_ap[2:, 1:-1] - lower_img_ap[:-2, 1:-1])/2**(i+1)
        dx = (lower_img_ap[1:-1, 2:] - lower_img_ap[1:-1, :-2])/2**(i+1)
        pylamid.append([dx, dy])
        logger.info(f"{dx.shape=}, {dy.shape=}")
        logger.info(f"{lower_img_ap.shape=}")
        logger.info(f"{lower_img[:-1,].shape=}")
        cv2.imwrite(f"img/{d=}_test.jpg", lower_img)
        lower_img = cv2.pyrDown(lower_img)

    alpha, beta = 0.01, 0.85
    next_factor = np.ones_like(pylamid[-1][0])
    epsilon = 0.00000001
    for i, H in enumerate(pylamid[::-1]):
        Hx, Hy = H[0], H[1]
        norm = np.sqrt(Hx**2 + Hy**2) + epsilon
        scale_factor = alpha / norm * (norm / alpha)**beta
        logger.info(f"{scale_factor.shape=}")
        if i == 0:
            next_factor = scale_factor * next_factor
        else:
            next_factor = scale_factor * upsampling(next_factor)

    H = pylamid[0]
    Gx, Gy = next_factor*H[0], next_factor*H[1]
    dGx, dGy = cal_divG(Gx, Gy)

    # 端だけの残した画像の生成
    new_img = copy.deepcopy(higres_img)
    new_img[1:-1, 1:-1, :] = 0

    tmpdGx = dGx[..., 0]
    h, w = tmpdGx.shape
    # ガウスザイデル法で利用する行列の生成
    coeff_x, coeff_y = get_coeff(h, w)
    A = coeff_x + coeff_y
    D, L, U = np.diag(A), np.tril(A, k=1), np.triu(A, k=1)
    M = - np.dot(np.linalg.inv(D+L), U)
    N = np.linalg.inv(D+L)
    # ガウスザイデル法の初期値を生成
    blur_img = higres_img
    for i in range(4):
        blur_img = cv2.blur(blur_img, (5, 5))
    blur_img = blur_img[1:-1, 1:-1]

    for i in range(3):
        b = get_b(new_img[..., i], dGx[..., i], dGy[..., i])
        x = blur_img[..., i].reshape((-1, 1))
        ans = gauss_seidel(M, N, b, 1000, epsilon, )
        ans.reshape((h, w))
        new_img[1:-1, 1:-1, i] = ans

    new_img = np.log(new_img)
    new_img = np.clip(255*new_img, 0, 1)
    new_img = np.uint8(new_img)
    cv2.imwrite("img/new_img.jpg", new_img)
