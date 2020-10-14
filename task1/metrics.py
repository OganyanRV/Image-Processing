import numpy
import math


def immse(fi_image: numpy.ndarray, se_image: numpy.ndarray) -> float:
    return ((fi_image - se_image) ** 2).mean()


def psnr(fi_image: numpy.ndarray, se_image: numpy.ndarray) -> float:
    mse = immse(fi_image, se_image)
    if mse == 0:
        return 100.0
    return 10 * math.log((255.0 * 255.0 / mse), 10)


def ssim(fi_image: numpy.ndarray, se_image: numpy.ndarray) -> float:
    fi_image.flatten()
    se_image.flatten()
    intensity_fi = fi_image.mean()  # Математическое ожидание (среднее арифметическое)
    intensity_se = se_image.mean()  # Математическое ожидание (среднее арифметическое)
    contrast_fi = fi_image.var()  # Среднеквадратическое отклонение (дисперсия)
    contrast_fi = contrast_fi ** 0.5
    contrast_se = se_image.var()  # Среднеквадратическое отклонение (дисперсия)
    contrast_se = contrast_se ** 0.5
    const_fi = 0.0003
    const_se = 0.0005
    koef_l = (2 * intensity_fi * intensity_se + const_fi) / \
        (intensity_fi ** 2 + intensity_se ** 2 + const_fi)
    if fi_image.shape == 2:
        covariance = numpy.cov(fi_image, se_image)  # Ковариация двух случайных величин
        koef_c = (2 * covariance + const_se) / \
            (contrast_fi ** 2 + contrast_se ** 2 + const_se)
        return (koef_l/koef_c).mean()
    koef_c = (2 * contrast_se * contrast_fi + const_se) / \
        (contrast_fi ** 2 + contrast_se ** 2 + const_se)
    return koef_l * koef_c
