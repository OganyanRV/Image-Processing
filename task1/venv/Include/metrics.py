import cv2
import numpy
import math
from Pillow import Image


def immse(fi_image:numpy.ndarray, se_image:numpy.ndarray) -> float:
    return ((fi_image - se_image)**2).mean()

def psnr(fi_image:numpy.ndarray, se_image:numpy.ndarray) -> float:
    mse = immse(fi_image, se_image)
    if mse==0:
        mse=255.0
    return 10 * math.log((255.0/mse),10)

def ssim(fi_image, se_image)-> float:
