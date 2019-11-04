# -*- coding: utf-8 -*-

import os
import cv2
import math
import librosa
import numpy as np
from __init__ import detector, predictor, vfps, size
from __init__ import ref_dir

def test_aloud(mp3_path, thr1=1e-3, thr2=0.5, ksize=3, n=100):
    # test if each synthesized frame is 'aloud' using threshold $(thr1)
    data, _ = librosa.load(mp3_path, sr=vfps*n)
    nfr = math.ceil(data.shape[0] / n)
    A = []
    for i in range(nfr):
        wave = np.array(data[i*n:min(i*n+n, data.shape[0])])
        volume = np.mean(wave ** 2)
        A.append(0 if volume < thr1 else 1)
    A = np.array(A, dtype=np.uint8)
    
    # connect 0-gaps with less than $(thr2) seconds
    span = int(thr2 * vfps + 1)
    i, j = 0, 0
    iszero = False
    for j in range(nfr):
        if A[j] == 0 and not iszero:
            i = j
            iszero = True            
        elif A[j] == 1 and iszero:
            iszero = False
            A[i:j] = 1 if j-i < span else A[i:j]
    if A[j] == 0 and j-i < span:
        A[i:j] = 1
            
    # apply dilation and erosion (closing operation)
    kernel = np.ones((ksize,), dtype=np.uint8)
    A2D = np.array([A]*A.shape[0])
    closing = cv2.morphologyEx(A2D, cv2.MORPH_CLOSE, kernel)
    A = closing[0, :]
    return A
    
def add_blink(eye_ldmk, thr):
    left  = cv2.contourArea(eye_ldmk[:6, :])
    right = cv2.contourArea(eye_ldmk[6:, :])
    return 1 if (left + right)/2 < thr else 0

def test_motion(tar_path, aB=1, blink_thr=0.025, ksize=3):
    cap = cv2.VideoCapture(tar_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    endfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cnt = 0
    FS, B = [], []
    while cap.isOpened():
        if cnt == endfr:
            break
        cnt += 1
        print('%s: %04d/%04d' % (tar_path, cnt, endfr))
        
        # get landmarks
        _, frame = cap.read()
        det = detector(frame, 1)[0]
        shape = predictor(frame, det)
        ldmk = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)], np.float32)
        
        # get face landmarks and eye landmarks and normalize them
        face, eye = ldmk[0:36], ldmk[36:48]
        scale = np.linalg.norm(np.mean(eye[:6, :], axis=0)-np.mean(eye[6:, :], axis=0), ord=2)
        face /= scale
        eye  /= scale
        blink = add_blink(eye, blink_thr)
        
        FS.append(face)
        B.append(blink)
        
    FS = np.array(FS)   
    B  = np.array(B)
    
    # apply dilation and erosion (closing operation)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    B2D = np.array([B]*B.shape[0], dtype=np.uint8)
    closing = cv2.morphologyEx(B2D, cv2.MORPH_CLOSE, kernel)
    B = closing[0, :]
    
    # first derivative of landmark positions
    FS = np.concatenate([FS[0][np.newaxis, :], FS, FS[-1][np.newaxis]], axis=0)
    V = np.linalg.norm(np.mean((FS[2:]-FS[:-2])/2, axis=1), axis=1)
    V += aB * B
    return V

def match_penalty(A, V, aU=2):
    G = np.zeros((A.shape[0], V.shape[0]))
    for i in range(G.shape[0]):
        term1 = V if A[i] == 0 else np.zeros(V.shape)
        term2 = aU*V if i >= 3 and A[i-2] == 1 and A[i-3] == 0 else np.zeros(V.shape)
        G[i, :] = term1 - term2
    return G

def preprocess(mp3_path, tar_path, npz_path):
    A = test_aloud(mp3_path)
    V = test_motion(tar_path)
    G = match_penalty(A, V)
    np.savez(npz_path, A=A, V=V, G=G)
    
def timing_opt(mp3_path, tar_path, tmp_path=None, preproc=False, aS=2):
    # check if preprocessing is needed
    _, fname = os.path.split(mp3_path)
    mp3_id, _ = os.path.splitext(fname)
    _, fname = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(fname)
    tmp_path = '%s%s-x%s' % (ref_dir, mp3_id, tar_id) if tmp_path is None else tmp_path
    if preproc or os.path.exists(tmp_path) == False:
        preprocess(mp3_path, tar_path, tmp_path)
    
    # initialize data
    data = np.load(tmp_path)
    A, V, G = data['A'], data['V'], data['G']
    N, M = G.shape
    F = np.array([np.inf]*N*M*2).reshape(N, M, 2)
    F[0, :, 0] = V if A[0] == 0 else 0
    
    # dynamic programming
    for i in range(1, N):
        if i % 100 == 0 or i == N-1:
            print('dynamic programming: %04d/%04d' % (i, N-1))
        for j in range(1, M):
            F[i, j, 0] = min(F[i-1, j-1, 0], F[i-1, j-1, 1]) + G[i, j]
            F[i, j, 1] = F[i-1, j, 0] + aS*V[j] + G[i, j]
    
    # optimize and backtrack
    _, js, ks = np.where(F == np.min(F[N-1, :, :]))
    assert(len(js) == 1 and len(ks) == 1)
    j, k = js[0], ks[0]
    L = np.zeros((N,))
    for i in range(N-1, -1, -1):
        # from N-1 to 0
        L[i] = j
        if i > 0:
            if k == 0:
                k = 0 if F[i-1, j-1, 0] < F[i-1, j-1, 1] else 1
                j -= 1
            else:
                k -= 1
    return L

def warp(prevfr, nextfr):
    pass
    
def retiming(tar_path, save_path, L):
    cap = cv2.VideoCapture(tar_path)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), vfps, size)
    idx = 0
    print('Start preprocessing...')
    while cap.isOpened():
        if idx == L.shape[0]:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, L[idx])
        _, frame = cap.read()
        writer.write(frame)
        
        idx += 1
        if idx % 100 == 0 or idx == L.shape[0]:
            print("%s: %04d/%04d" % (save_path, idx, L.shape[0]))
    print('Done')
    
def run():
    mp3_path = 'input/test036.mp3'
    tar_path = 'target/target001.mp4'
    tmp_path = 'reference/retiming_t36i1.npz'
    save_path = 'target/rt_target001.mp4'
    L = timing_opt(mp3_path, tar_path, tmp_path)
#    np.save('reference/L.npy', L)
    print('Start from frame %04d.' % L[0])
    retiming(tar_path, save_path, L)    

if __name__ == '__main__':
    run()
