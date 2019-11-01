# -*- coding: utf-8 -*-

import cv2
import math
import librosa
import numpy as np
from __init__ import vfps

def test_aloud(mp3_path, thr1=1e-3, thr2=0.5, ksize=3, n=100):
    # test if each synthesized frame is 'aloud' using threshold $(thr1)
    data, _ = librosa.load(mp3_path, sr=vfps*n)
    nfr = math.ceil(data.shape[0] / n)
    A = []
    for i in range(nfr):
        wave = np.array(data[i*n:min(i*n+n, data.shape[0])])
        volume = np.mean(wave ** 2)
        A.append(0 if volume < thr1 else 1)
    A = np.array(A)
    
    # connect 0-gaps with less than $(thr2) seconds
    span = int(thr2 * vfps + 1)
    for i in range(nfr-span):
        if A[i] == 1 and A[i+span-1] == 1:
            A[i:i+span] = 1
            
    # apply dilation and erosion
    kernel = np.ones((ksize,), dtype=np.uint8)
#    A = cv2.dilate(A, kernel)
#    A = cv2.erode(A, kernel)
    
    return A

def test_blink():
    pass

if __name__ == '__main__':
    A = test_aloud('input/test036.mp3')
    print('Hello, World')