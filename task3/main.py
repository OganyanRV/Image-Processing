import cv2
import numpy as np


def multiply_brightness_by(image, mult):
    return np.array([[[np.uint8(min(255, i*mult)) for i in x] for x in y] for y in image])


def basic_action():
    img = cv2.imread("task1/first_img.jpg")
    cv2.imshow('original image', img)
    img2 = multiply_brightness_by(img, 1.5)
    cv2.imshow('brightnes increased by 1.5', img2)
    img3 = multiply_brightness_by(img, 0.667)
    cv2.imshow('brightnes decreased by 1.5', img3)
    print(img[0][0])
    print(img2[0][0])
    print(img3[0][0])
    cv2.waitKey(0)


if __name__ == "__main__":
    basic_action()
