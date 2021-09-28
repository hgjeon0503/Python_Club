# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:48:25 2021

@author: user
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

celeberity_images_fp = "D:/similarity_report/celeberities"

my_face = cv2.imread("D:/similarity_report/my_face/test.jpg", cv2.IMREAD_GRAYSCALE)
my_face = cv2.resize(my_face, dsize=(300, 400), interpolation=cv2.INTER_AREA)
# print(my_face)
factor = 0.85

celub_list = []
for celub in os.listdir(celeberity_images_fp):
    celub_img = None

    celub_img = cv2.imread(celeberity_images_fp+"/"+celub, cv2.IMREAD_GRAYSCALE)
    celub_img = cv2.resize(celub_img, dsize=(300, 400), interpolation=cv2.INTER_NEAREST)

    sift = cv2.xfeatures2d.SIFT_create()
    res = None

    kp1, des1 = sift.detectAndCompute(celub_img, None)
    kp2, des2 = sift.detectAndCompute(my_face, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append([m])

    result = len(good)
    celub_list.append(result)
    print(celub, result)

most_similar = np.argmax(celub_list)

most_similar_celub = os.listdir(celeberity_images_fp)[most_similar]
most_similar_celub = cv2.imread(celeberity_images_fp+"/"+most_similar_celub, cv2.IMREAD_GRAYSCALE)
most_similar_celub = cv2.resize(most_similar_celub, dsize=(300, 400), interpolation=cv2.INTER_AREA)

kp1, des1 = sift.detectAndCompute(most_similar_celub, None)
kp2, des2 = sift.detectAndCompute(my_face, None)
report = cv2.drawMatchesKnn(my_face,kp1,most_similar_celub,kp2,good,None,flags=2)

plt.imshow(report)
plt.title("most similiar")
plt.show()

# not_similar = np.argmin(celub_list)
#
# not_similar_celub = os.listdir(celeberity_images_fp)[not_similar]
# not_similar_celub = cv2.imread(celeberity_images_fp+"/"+not_similar_celub, cv2.IMREAD_GRAYSCALE)
# not_similar_celub = cv2.resize(not_similar_celub, dsize=(300, 400), interpolation=cv2.INTER_AREA)
#
# kp1, des1 = sift.detectAndCompute(not_similar_celub, None)
# kp2, des2 = sift.detectAndCompute(my_face, None)
# report = cv2.drawMatchesKnn(my_face,kp1,not_similar_celub,kp2,good,None,flags=2)
#
# plt.imshow(report)
# plt.title("not similiar")
# plt.show()


