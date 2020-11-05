import cv2
import numpy as np
from task1.metrics import ssim
from utils.utils import measure_time


def multiply_brightness_by(image: np.ndarray, mult: float):
    assert mult >= 0, "multiplier must be positive"
    return np.array([[[np.uint8(min(255, i*mult)) for i in x] for x in y] for y in image])


def multiply_hsv_brightness_by(image: np.ndarray, mult: float):
    assert mult >= 0, "multiplier must be positive"
    return np.array([[[x[0], x[1], np.uint8(min(255, x[2]*mult))] for x in y] for y in image])


def basic_action():

    img = cv2.imread("task1/first_img.jpg")
    cv2.imshow('original image', img)
    time2, img2 = measure_time(multiply_brightness_by, img, 1.5)
    cv2.imshow('brightnes multiplied by 1.5', img2)
    time3, img3 = measure_time(multiply_brightness_by, img, 1/1.5)
    cv2.imshow('brightnes divided by 1.5', img3)
    print('Brightness multiplying of RGB (BGR) took', time2, 'and', time3, 'seconds')

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    time2, hsv_img2 = measure_time(multiply_hsv_brightness_by, hsv_img, 1.5)
    time3, hsv_img3 = measure_time(multiply_hsv_brightness_by, hsv_img, 1/1.5)
    print('Brightness multiplying of HSV took', time2, 'and', time3, 'seconds')
    print('structure similarity index for BGR * 1.5 and HSV * 1.5 is:',
          ssim(cv2.cvtColor(hsv_img2, cv2.COLOR_HSV2BGR), img2))
    print('structure similarity index for BGR / 1.5 and HSV / 1.5 is:',
          ssim(cv2.cvtColor(hsv_img3, cv2.COLOR_HSV2BGR), img3))
    cv2.waitKey(0)


if __name__ == "__main__":
    basic_action()
