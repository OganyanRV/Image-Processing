import cv2 as cv
import numpy as np
from task31.convert_to_hsv_lib import convert_to_hsv
from task1.metrics import psnr
from utils.utils import measure_time


def main():
    # Загружаем изображение из памяти
    img = cv.imread('house.jpeg', cv.IMREAD_COLOR)

    # Запускаем таймер и производим конвертацию нашей реализацией
    our_time, converted = measure_time(convert_to_hsv, img)

    converted = np.array(converted)

    print('Время работы нашей реализации: {0} с'.format(our_time))

    # Запускаем таймер и производим конверацию встроенной функцией
    built_in_time, built_in = measure_time(cv.cvtColor, img, cv.COLOR_BGR2HSV)

    print('Время работы библиотечной реализации: {0} с'.format(built_in_time))

    print('Метрика PSNR: {0}'.format(psnr(converted, built_in)))

    # Преобразум тип в int16, чтобы посчитать максимальную разницу в значениях
    converted = converted.astype(np.int16, copy=False)
    built_in = built_in.astype(np.int16, copy=False)

    # Вычисляем наибольшую разницу
    print('Максимальная разница в значениях: {0}'.format(np.amax(np.absolute(converted - built_in))))


if __name__ == '__main__':
    main()
