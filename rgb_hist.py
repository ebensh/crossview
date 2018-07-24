#!/usr/bin/env python3

import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]

img = cv2.imread(path)
color = ('b','g','r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
#plt.show()
plt.savefig(path + "_rgb_hist.png")