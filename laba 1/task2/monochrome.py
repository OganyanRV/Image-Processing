import numpy


def monochrome(source_image: numpy.ndarray) -> numpy.ndarray:
    new_image = source_image
    b_comp = source_image[:, :, 0]
    g_comp = source_image[:, :, 1]
    r_comp = source_image[:, :, 2]
    new_comp = (b_comp + g_comp + r_comp) / 3
    new_image[:, :, 0] = new_comp
    new_image[:, :, 1] = new_comp
    new_image[:, :, 2] = new_comp
    return new_image


def transform(source_image: numpy.ndarray) -> numpy.ndarray:
    new_image = numpy.zeros((source_image.shape[0], source_image.shape[1]))
    for i in range(0, source_image.shape[0]):
        for j in range(0, source_image.shape[1]):
            new_image[i][j] = source_image[i][j][0]
    return new_image
