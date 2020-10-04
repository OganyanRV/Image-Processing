import cv2
import numpy
import math
from PIL import Image


def immse(fi_image:numpy.ndarray, se_image:numpy.ndarray) -> float:
    return ((fi_image - se_image)**2).mean()

def psnr(fi_image:numpy.ndarray, se_image:numpy.ndarray) -> float:
    mse = immse(fi_image, se_image)
    if mse==0:
        mse=255.0
    return 10 * math.log((255.0/mse),10)

def ssim(fi_image:numpy.ndarray, se_image:numpy.ndarray)-> float:
    intensity_fi = fi_image.mean() # Математическое ожидание (среднее арифметическое)
    intensity_se = se_image.mean() # Математическое ожидание (среднее арифметическое)
    contrast_fi = fi_image.var() # Среднеквадратическое отклонение (дисперсия)
    contrast_fi = sqrt(contrast_fi)
    contrast_se = se_image.var() # Среднеквадратическое отклонение (дисперсия)
    contrast_se = sqrt(contrast_se)
    covariance = numpy.cov(contrast_fi, contrast_se) # Ковариация двух случайных величин
    const_fi = 0.0003
    const_se = 0.0005
    return ((2*intensity_fi*intensity_se + const_fi) * (2*covariance + const_se )) /  ((intensity_fi**2 + intensity_se**2 + const_fi) *( contrast_fi**2 +  contrast_se ** 2 +const_se ))


