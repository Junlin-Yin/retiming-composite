# -*- coding: utf-8 -*-

import cv2
from __init__ import Square, predictor, detector, vfps, size
from facefrontal import facefrontal, warp_mapping
import numpy as np

padw = 95
detw = 130

def getGaussianPyr(img, layers):
    g = img.astype(np.float64)
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
        pyramid.append((gi - gi_aprx))
    return pyramid[::-1]

def reconstruct(G, Lappyr):
    for i in range(len(Lappyr)-1, -1, -1):
        # len(Gaupyr)-1, ..., 1, 0
        G = cv2.pyrUp(G)
        G = cv2.resize(G, Lappyr[i].shape[:2][::-1])
        G += Lappyr[i]
    return G.astype(np.uint8)

def pyramid_blend(img1, img2, mask_, layers=4):
    assert(img1.shape == img2.shape and img1.shape[:2] == mask_.shape)
    mask = mask_ / np.max(mask_)    # 0 ~ 1
    # construct Gaussian pyramids of input images
    Gaupyr1 = getGaussianPyr(img1, layers+1)
    Gaupyr2 = getGaussianPyr(img2, layers+1)
    Gaupyrm = getGaussianPyr(mask, layers+1)
    
    # construct Laplacian pyramids of input images
    Lappyr1 = getLaplacianPyr(Gaupyr1)
    Lappyr2 = getLaplacianPyr(Gaupyr2)
    
    # blend pyramids in every layer
    Gaupyrm1 = Gaupyrm[:-1]
    Gaupyrm2 = [1-msk for msk in Gaupyrm1]
    BLappyr1 = [lap * msk[:, :, np.newaxis] for lap, msk in zip(Lappyr1, Gaupyrm1)]
    BLappyr2 = [lap * msk[:, :, np.newaxis] for lap, msk in zip(Lappyr2, Gaupyrm2)]
    BLappyr  = [lap1 + lap2 for lap1, lap2 in zip(BLappyr1, BLappyr2)]
    initG = Gaupyr1[-1] * Gaupyrm[-1][:, :, np.newaxis] + Gaupyr2[-1] * (1-Gaupyrm[-1])[:, :, np.newaxis]
    
    # collapse pyramids and form the blended image
    img = reconstruct(initG, BLappyr)
    return img

def getindices(ftl_face, sq, padw=padw, detw=detw):
    # get mask region using boundary, chin landmarks and nose landmarks
    # boundary region: left -> right, upper -> lower
    WH = ftl_face.shape[0]
    boundary = sq.align(detw)
    left, right, upper, lower = np.array(boundary) + padw
    indices = np.array([(x, y) for x in range(left, right) for y in range(upper, lower)])
    
    # get landmarks of frontalized face
    det = detector(ftl_face, 1)[0]
    shape = predictor(ftl_face, det)
    ldmk = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
    chin_xp, chin_fp = ldmk[ 3:14, 0], ldmk[ 3:14, 1]
    chin_line = np.interp(np.arange(WH), chin_xp, chin_fp)
    nose_xp, nose_fp = ldmk[31:36, 0], ldmk[31:36, 1]
    nose_line = np.interp(np.arange(WH), nose_xp, nose_fp)

    # filter the position which is out of chin line and nose line    
    check = np.logical_and(indices[:, 1] < chin_line[indices[:, 0]],
                           indices[:, 1] > nose_line[indices[:, 0]])
    return indices[check.nonzero()]

def align2target(syntxtr, tar_shape, sq, padw=padw, detw=detw):
    # align lower-face to target frame
#   |padw| detw  |padw|
#   |----|-------|---------
#   |                 |padw
#   |    ---------    -----
#   |    |       |    |
#   |    |       |    |detw
#   |    |       |    |
#   |    ---------    -----
#   |     ftl_face    |padw
#   -----------------------
    rsize = sq.getrsize(syntxtr.shape)
    syn_face_ = np.zeros((rsize, rsize, syntxtr.shape[2]), dtype=np.uint8)
    left, right, upper, lower = sq.align(rsize)
    syn_face_[upper:lower, left:right, :] = syntxtr
    syn_face_ = cv2.resize(syn_face_, (detw, detw))
    syn_face = np.zeros(tar_shape, dtype=np.uint8)
    syn_face[padw:padw+detw, padw:padw+detw, :] = syn_face_
    return syn_face

def recalc_pixel(pt, coords, pixels, thr=5, sigma=0.2):
    L2 = np.linalg.norm(coords-pt, ord=2, axis=1)
    indx  = np.where(L2 <= thr)
    weights = np.exp(-L2[indx]**2 / (2* sigma**2))
    weights /= np.sum(weights)  # np.sum(weights) == 1
    return np.matmul(weights, pixels[indx, :])

def warpback(face, tarfr, tarldmk, indices, projM, transM):
    # get the pixels of given indices
    pixels = face[indices[:, 1], indices[:, 0], :]           # (N, 3)
    
    # get the to-be-recalculated region in the original frame
    warp_mask, region, coords, pixels = warp_mapping(indices, pixels, tarfr, tarldmk, projM, transM)
    
    # do recalculation for every pixel in the region
    tmpfr = np.zeros(tarfr.shape, dtype=np.uint8)
    for pt in region:
        tmpfr[pt[1], pt[0], :] = recalc_pixel(pt, coords, pixels)
    tmpfr = cv2.inpaint(tmpfr, ~warp_mask, 10, cv2.INPAINT_TELEA)
    return pyramid_blend(tmpfr, tarfr, warp_mask)

def synthesize_frame(tarfr, syntxtr, sq):
    # frontalize the target frame
    ftl_face, ldmk, projM, transM = facefrontal(tarfr, detector, predictor, detail=True)
    
    # align lower-face to target frame
    syn_face = align2target(syntxtr, ftl_face.shape, sq)
   
    # get indices of pixels in ftl_face which needs to be blended into target frame
    indices = getindices(ftl_face, sq)
    
    # warp the synthesized face to the original pose and blending
    return warpback(syn_face, tarfr, ldmk, indices, projM, transM)

def composite(inp_path, tar_path, save_path, sq):
    syndata = np.load(inp_path)
    cap = cv2.VideoCapture(tar_path)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), vfps, size)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert(syndata.shape[0] == nfr)
    for i in range(nfr):
        print('%s: %04d/%04d' % (save_path, i+1, nfr))
        ret, tarfr = cap.read()
        assert(ret)
        frame = synthesize_frame(tarfr, syndata[i], sq)
        writer.write(frame)
    print('%s: synthesis done.' % save_path)
    return save_path

def test1():
    left = cv2.imread('tmp/left.png')
    right = cv2.imread('tmp/right.png')
    mask = np.zeros(left.shape)
    mask[:, :mask.shape[1]//2, :] = 1
    n = 6
    spec = np.zeros((mask.shape[0]*n, mask.shape[1], mask.shape[2]))
    for layers in range(n):
        blend = pyramid_blend(left, right, mask, layers)
        spec[mask.shape[0]*layers:mask.shape[0]*(layers+1), :, :] = blend
    cv2.imwrite('reference/blend.png', spec)
    
def test2():
    tarfr = cv2.imread('tmp/0660.png')
    region = np.load('tmp/region.npy')
    tarfr[region[:, 1], region[:, 0], :] = (255, 255, 0)
    cv2.imwrite('tmp/regiontest.png', tarfr)
    
def test3():
    tarfr = cv2.imread('tmp/0660.png')
    syntxtr = cv2.imread('tmp/syn100.png')
    sq = Square(0.25, 0.75, 0.6, 1.0)
    outpfr = synthesize_frame(tarfr, syntxtr, sq)
    cv2.imwrite('tmp/i100t660.png', outpfr)

if __name__ == '__main__':
    test3()