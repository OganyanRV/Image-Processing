import cv2
import time
from task2.monochrome import *
from task1.metrics import *

image = cv2.imread("source_img.jpg")
cv2.imshow('Original image', image)

print("Время работы моей конвертации в монохромное", end="\n")
start = time.time()
new_my_image = monochrome(image)
end = time.time()
print(end-start, end="\n")
cv2.imshow('My monochrome image', new_my_image)

print("Время работы встроенной конвертации в монохромное", end="\n")
start = time.time()
new_opencv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2GRAY)
end = time.time()
print(end-start, end="\n")
cv2.imshow('Opencv monochrome image', new_opencv_image)

print("Сравнение картинок:", end="\n")
new_my_image = transform(new_my_image)
print("IMMSE:", end=" ")
print(immse(new_my_image, new_opencv_image), end="\n")
print("PSNR:", end=" ")
print(psnr(new_my_image, new_opencv_image), end="\n")
print("SSIM:", end=" ")
print(ssim(new_my_image, new_opencv_image), end="\n")
cv2.waitKey(0)
