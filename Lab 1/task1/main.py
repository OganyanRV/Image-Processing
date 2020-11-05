import cv2
import warnings
from skimage.util import random_noise
from skimage.metrics import structural_similarity
from task1.metrics import *

warnings.filterwarnings("ignore", category=UserWarning)

image_first = cv2.imread("first_img.jpg")
cv2.imwrite('sourceimage.jpg', image_first)

image_second = cv2.cvtColor(image_first, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', image_second)

noise_img_fi = random_noise(image_first, mode='s&p', amount=0.1)  # Добавление шума
cv2.imshow('noised source image', noise_img_fi)

noise_img_se = random_noise(image_second, mode='s&p', amount=0.1)  # Добавление шума
cv2.imshow('noised gray  image', noise_img_se)
cv2.waitKey(0)

print("IMMSE одинаковых ", end="\n")
print(immse(image_first, image_first), end="\n")

print("PSNR одинаковых ", end="\n")
print(psnr(image_first, image_first), end="\n")

print("SSIM одинаковых ", end="\n")
print(ssim(image_first, image_first), end="\n")

print("IMMSE цветного ", end="\n")
print(immse(image_first, noise_img_fi), end="\n")

print("IMMSE серого ", end="\n")
print(immse(image_second, noise_img_se), end="\n")

print("PSNR цветного ", end="\n")
print(psnr(image_first, noise_img_fi), end="\n")

print("PSNR серого: ", end="\n")
print(psnr(image_second, noise_img_se), end="\n")

print("SSIM цветных: ", end="\n")
print("Мой SSIM: ", end="\n")
print('{:.10f}'.format(ssim(image_first, noise_img_fi)), end="\n")

print("structural_similarity: ", end="\n")
print(structural_similarity(image_first, noise_img_fi, multichannel=True), end="\n")

print("SSIM серых: ", end="\n")
print("Мой SSIM: ", end="\n")
print('{:.10f}'.format(ssim(image_second, noise_img_se)), end="\n")

print("structural_similarity: ", end="\n")
print(structural_similarity(image_second, noise_img_se, multichannel=True), end="\n")
