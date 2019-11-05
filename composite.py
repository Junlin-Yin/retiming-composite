# -*- coding: utf-8 -*-

import cv2
import numpy as np

def getGaussianPyr(img, layers):
    g = img
    pyramid = [g]
    for i in range(layers):
        g = cv2.pyrDown(g)
        pyramid.append(g)
    return pyramid

def getLaplacianPyr(Gaupyr):
    pyramid = []
    for i in range(len(Gaupyr)-2, -1, -1):
        # len(Gaupyr)-2, ..., 1, 0
        gi = Gaupyr[i]
        gi_aprx = cv2.pyrUp(Gaupyr[i+1])
        gi_aprx = cv2.resize(gi_aprx, gi.shape[:2][::-1])
        pyramid.append(gi - gi_aprx)
    return pyramid[::-1]

def reconstruct(G, Lappyr):
    for i in range(len(Lappyr)-1, -1, -1):
        # len(Gaupyr)-1, ..., 1, 0
        G = cv2.pyrUp(G)
        G = cv2.resize(G, Lappyr[i][:2][::-1])
        G += Lappyr[i]
    return G

def pyramid_blend(img1, img2, mask, layers):
    # construct Gaussian pyramids of input images
    Gaupyr1 = getGaussianPyr(img1, layers+1)
    Gaupyr2 = getGaussianPyr(img2, layers+1)
    Gaupyrm = getGaussianPyr(mask, layers+1)
    
    # construct Laplacian pyramids of input images
    Lappyr1 = getLaplacianPyr(Gaupyr1)
    Lappyr2 = getLaplacianPyr(Gaupyr2)
    
    # blend pyramids in every layer
    Gaupyrm1 = Gaupyrm[:-1]
    Gaupyrm2 = [255-msk for msk in Gaupyrm1]
    BLappyr1 = [lap & msk for lap, msk in zip(Lappyr1, Gaupyrm1)]
    BLappyr2 = [lap & msk for lap, msk in zip(Lappyr2, Gaupyrm2)]
    BLappyr  = [lap1 + lap2 for lap1, lap2 in zip(BLappyr1, BLappyr2)]
    initG = Gaupyr1[-1] & Gaupyrm[-1] + Gaupyr2[-1] & (255-Gaupyrm[-1])
    
    # collapse pyramids and form the blended image
    img = reconstruct(initG, BLappyr)
    return img

def test1():
    pass

if __name__ == '__main__':
    print('Hello, World')