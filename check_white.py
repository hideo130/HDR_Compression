# import cv2
import numpy as np
import PySimpleGUI as sg
from logging import getLogger, INFO, basicConfig
import io
from PIL import Image


def check_white(img, ths):
    new_img = np.zeros_like(img)
    for i, th in enumerate(ths):
        ch = img[:, :, i]
        if i == 0:
            new_img[:, :, i] = np.where(ch > th, 255, ch)
        else:
            new_img[:, :, i] = np.where(ch > th, 0, ch)
    return new_img
# cv2.imwrite("img/test.jpg", lower_img)


do_debug = True
if do_debug:
    basicConfig(level=INFO)
logger = getLogger(__name__)


def call_img(img):
    """
    input img:Image object
    return img:画像のbytes型
    sg.ImageはPNGかGIFしか読み込むことができない．
    レンダリングしたpng画像やハイパースペクトル画像のpng形式のbytes型を作成し，
    その後，sg.Imageの引数dataに渡すことで画像を表示している．
    """
    bio = io.BytesIO()
    # バッファに画像を出力
    img.save(bio, format="PNG")
    # cv2.imwrite("tmp.png", img)
    # バッファの全内容の bytes型 をgetvalue()で取得する
    img = bio.getvalue()
    return img


# img = cv2.imread('img/sample.jpg')
img = Image.open('img/d=3_test.jpg')

show_img = call_img(img)
img = np.array(img)
checked_img = check_white(img, [250, 250, 250])
checked_img = Image.fromarray(checked_img)
checked_img = call_img(checked_img)


layout = [
    [sg.Slider(range=(0, 254), orientation='h', enable_events=True,
               size=(34, 20), key='__SLIDER1__', default_value=250)],
    # [sg.Slider(range=(1, 255), orientation='h', enable_events=True,
    #            size=(34, 20), key='__SLIDER2__', default_value=200)],
    [sg.Image(data=show_img)], [sg.Image(data=checked_img,  key='_OUTPUT_')]]
window = sg.Window('白飛びチェック', layout, finalize=True)

while True:
    event, values = window.read()
    logger.info(event)
    if event in (None, 'Quit'):
        print('exit')
        break
    elif event == "__SLIDER1__":
        logger.info(values)
        th = int(values["__SLIDER1__"])
        logger.info("update edge")
        checked_img = check_white(img, [th, th, th])
        checked_img = Image.fromarray(checked_img)
        checked_img = call_img(checked_img)
        window['_OUTPUT_'].update(data=checked_img)
    # elif event == "__SLIDER2__":
    #     logger.info(values)
    #     thmax = int(values["__SLIDER2__"])
    #     logger.info("update edge")

window.close()
