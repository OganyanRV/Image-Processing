import numpy
import cv2

def monochrome(source_image: numpy.ndarray, mode: int = 0) -> numpy.ndarray:

    new_image = source_image

    b_comp = source_image[:, :, 0]
    g_comp = source_image[:, :, 1]
    r_comp = source_image[:, :, 2]

    new_comp = (b_comp + g_comp + r_comp) / 3
# Надо сделать так, чтобы 3 компоненты он записал в 1
    if mode==0:
        numpy.reshape(new_image, 3)
        new_image[:, :, :] = new_comp
        return new_image

    new_image[:, :, 0] = new_comp
    new_image[:, :, 1] = new_comp
    new_image[:, :, 2] = new_comp

    return new_image


