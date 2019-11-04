# -*- coding: utf-8 -*-

import os
import cv2
import math
import librosa
import numpy as np
from __init__ import detector, predictor, vfps, size
from __init__ import ref_dir, tar_dir

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
    
def timing_opt(mp3_path, tar_path, tmp_path, preproc=False, aS=2):
    # check if preprocessing is needed
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

def half_warp(prevfr, nextfr):
    prev_gray = cv2.cvtColor(prevfr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(nextfr, cv2.COLOR_BGR2GRAY)
    H, W = prev_gray.shape
    flow = np.zeros(prev_gray.shape)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, flow, 
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                                        poly_n=5, poly_sigma=1.2, flags=0)
    flow /= 2
    flow[:, :, 0] += np.arange(W)                           # x axis
    flow[:, :, 1] += np.arange(H)[:, np.newaxis]            # y axis
    resfr = cv2.remap(prevfr, flow, None, cv2.INTER_LINEAR) # interpolating
    return resfr
    
def match(tar_path, save_path, L, warp):
    # read in all frames of target video
    print('Start forming new target video...')
    cnt = 0
    cap = cv2.VideoCapture(tar_path)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, L[0])
    ret, firstfr = cap.read()
    assert(ret)
    writer.write(firstfr)
    cnt += 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, L[1])
    ret, prefr = cap.read()
    assert(ret)
    for i in range(2, L.shape[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, L[i])
        ret, curfr = cap.read()
        assert(ret)
        
        # check and warp the duplicate frames
        if warp and L[i-2] == L[i-1]:
            tmpfr1 = half_warp(prefr, curfr).astype(np.int)
            tmpfr2 = half_warp(curfr, prefr).astype(np.int)            
            prefr = ((tmpfr1 + tmpfr2) / 2).astype(np.uint8)
        
        writer.write(prefr)
        prefr = curfr
        cnt += 1        
        if cnt % 100 == 0:
            print('%s: %04d/%04d' % (save_path, i, L.shape[0]))
        
    writer.write(prefr)
    cnt += 1
    print('Done')

def retiming(mp3_path, tar_path, tmp_path=None, save_path=None, warp=True):
    _, fname = os.path.split(mp3_path)
    mp3_id, _ = os.path.splitext(fname)
    _, fname = os.path.split(tar_path)
    tar_id, _ = os.path.splitext(fname)
    tmp_path = '%s%s-x-%s.npz' % (ref_dir, mp3_id, tar_id) if tmp_path is None else tmp_path
    save_path = '%s%s-x-%s.mp4' % (tar_dir, mp3_id, tar_id) if save_path is None else save_path
    
    L = timing_opt(mp3_path, tar_path, tmp_path)
    match(tar_path, save_path, L, warp)
    return save_path, L[0]

def test1():
    # optical flow warping test
    prevfr = cv2.imread('tmp/777.png')
    nextfr = cv2.imread('tmp/778.png')
    resfr1 = half_warp(prevfr, nextfr)
    resfr2 = half_warp(nextfr, prevfr)
    resfr  = ((resfr1.astype(np.int) + resfr2.astype(np.int)) / 2).astype(np.uint8)
    spec = np.zeros((resfr.shape[0]*5, resfr.shape[1], resfr.shape[2]))
    spec[resfr.shape[0]*0:resfr.shape[0]*1, :, :] = prevfr
    spec[resfr.shape[0]*1:resfr.shape[0]*2, :, :] = resfr1
    spec[resfr.shape[0]*2:resfr.shape[0]*3, :, :] = resfr
    spec[resfr.shape[0]*3:resfr.shape[0]*4, :, :] = resfr2
    spec[resfr.shape[0]*4:resfr.shape[0]*5, :, :] = nextfr
    cv2.imwrite('reference/777-778_spec.png', spec)    

if __name__ == '__main__':
    print('Hello, World')
    