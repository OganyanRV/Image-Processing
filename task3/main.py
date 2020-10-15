import cv2
import numpy as np


def multiply_brightness_by(image: np.ndarray, mult: float):
    assert mult >= 0, "multiplier must be positive"
    return np.array([[[np.uint8(min(255, i*mult)) for i in x] for x in y] for y in image])


def multiply_hsv_brightness_by(image: np.ndarray, mult: float):
    assert mult >= 0, "multiplier must be positive"
    return np.array([[[x[0], x[1], np.uint8(min(255, x[2]*mult))] for x in y] for y in image])


def ssim(fi_image: np.ndarray, se_image: np.ndarray) -> float:
    fi_image.flatten()
    se_image.flatten()
    intensity_fi = fi_image.mean()
    intensity_se = se_image.mean()
    contrast_fi = fi_image.var()
    contrast_fi = contrast_fi ** 0.5
    contrast_se = se_image.var()
    contrast_se = contrast_se ** 0.5
    const_fi = 0.0003
    const_se = 0.0005
    koef_l = (2 * intensity_fi * intensity_se + const_fi) / \
        (intensity_fi ** 2 + intensity_se ** 2 + const_fi)
    if fi_image.shape == 2:
        covariance = np.cov(fi_image, se_image)
        koef_c = (2 * covariance + const_se) / \
            (contrast_fi ** 2 + contrast_se ** 2 + const_se)
        return (koef_l/koef_c).mean()
    koef_c = (2 * contrast_se * contrast_fi + const_se) / \
        (contrast_fi ** 2 + contrast_se ** 2 + const_se)
    return koef_l * koef_c


def basic_action():

    img = cv2.imread("task1/first_img.jpg")
    cv2.imshow('original image', img)
    img2 = multiply_brightness_by(img, 1.5)
    cv2.imshow('brightnes multiplied by 1.5', img2)
    img3 = multiply_brightness_by(img, 1/1.5)
    cv2.imshow('brightnes divided by 1.5', img3)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img2 = multiply_hsv_brightness_by(hsv_img, 1.5)
    hsv_img3 = multiply_hsv_brightness_by(hsv_img, 1/1.5)
    print('structure similarity index for BGR * 1.5 and HSV * 1.5 is:',
          ssim(cv2.cvtColor(hsv_img2, cv2.COLOR_HSV2BGR), img2))
    print('structure similarity index for BGR / 1.5 and HSV / 1.5 is:',
          ssim(cv2.cvtColor(hsv_img3, cv2.COLOR_HSV2BGR), img3))
    cv2.waitKey(0)


if __name__ == "__main__":
    basic_action()
