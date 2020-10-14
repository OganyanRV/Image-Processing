import numpy
import cv2
import timeit
import time
from monochrome import *
from Include.task1.metrics import *

image = cv2.imread("source_img.jpg")
cv2.imshow('Original image', image)

print("Время работы моей конвертации в монохромное", end="\n")
start = time.time()
new_my_image = monochrome(image)
end = time.time()
print(end-start, end="\n")
cv2.imshow('My monochrome image', new_my_image)
#print(timeit.timeit(monochrome(image), number = 1), end = "\n") # Не работает

print("Время работы встроенной конвертации в монохромное", end = "\n")
start = time.time()
new_opencv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2GRAY)
end = time.time()
print(end-start, end="\n")
cv2.imshow('Opencv monochrome image', new_opencv_image)
#print(timeit.timeit('cv2.cvtColor(image_first, cv2.COLOR_BGR2GRAY)', number = 1), end = "\n") # Не работает
print("Сравнение картинок:", end = "\n")
print("IMMSE:", end = " ")
print(immse(new_my_image, new_opencv_image), end = "\n")
print("PSNR:", end = " ")
print(psnr(new_my_image, new_opencv_image), end = "\n")
print("SSIM:", end = " ")
print(ssim(new_my_image, new_opencv_image), end = "\n")
cv2.waitKey(0)