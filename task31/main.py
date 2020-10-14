# from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from task31.convert_to_hsv_lib import convert_to_hsv
from task1.metrics import psnr
import timeit

def main():
    img = cv.imread('Small.jpg', cv.IMREAD_COLOR)

    # cv.imshow("Original", cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    print(psnr(img, img))

    converted: np.array = convert_to_hsv(img)
    built_in = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    threshold = 0

    print(np.allclose(converted, built_in, threshold))

    # res = converted - built_in
    # cv.imshow("Converted", cv.cvtColor(converted, cv.COLOR_HSV2RGB))
    # cv.waitKey(0)
    # cv.destroyAllWindows()



    print(psnr(converted, img))

    print("Время работы нашей реализации: ")

if __name__ == '__main__':
    main()