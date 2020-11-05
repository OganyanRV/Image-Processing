import numpy as np


def convert_to_hsv(img: np.array) -> np.array:
    tmp = img.tolist()

    for row in tmp:
        for pixel in row:

            # Выделяем занчение каждого цвета отдельно
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

            # Округляем веществнные числа до целых
            h = np.uint8(h / 360 * 179)
            s = np.uint8(s * 255)
            v = np.uint8(v * 255)

            pixel[0] = h
            pixel[1] = s
            pixel[2] = v

    return np.array(tmp)
