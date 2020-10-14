import cv2 as cv
import numpy as np


def convert_to_hsv(img: np.array) -> np.array:
    convert_function = np.vectorize(convert_per_pixel, signature="(n)->(n)")
    return convert_function(img)

def convert_per_pixel(pixel):
    pixel_blue: int = pixel[0] / 255
    pixel_green: int = pixel[1] / 255
    pixel_red: int = pixel[2] / 255

    cmax: int = max(pixel_blue, pixel_green, pixel_red)
    cmin: int = min(pixel_blue, pixel_green, pixel_red)

    if cmax - cmin == 0:
        h = 0
    elif abs(cmax - pixel_red) < 10 ** (-3):
        if pixel_green >= pixel_blue:
            h = 60 * (pixel_green - pixel_blue) / (cmax - cmin)
        else:
            h = 60 * (pixel_green - pixel_blue) / (cmax - cmin) + 360
    elif abs(cmax - pixel_green) < 10 ** (-3):
        h = 60 * (pixel_blue - pixel_red) / (cmax - cmin) + 120
    else:
        h = 60 * (pixel_red - pixel_green) / (cmax - cmin) + 240

    if abs(cmax) < 10 ** (-3):
        s = 0
    else:
        s = 1 - cmin / cmax

    v = cmax

    h = np.uint8(h / 360 * 179)
    s = np.uint8(s * 255)
    v = np.uint8(v * 255)

    return np.array([h, s, v])
