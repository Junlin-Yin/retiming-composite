# -*- coding: utf-8 -*-

import cv2
from __init__ import vfps, size

def util1(tar_path, save_path, startfr=0, endfr=None):
    # clip target video
    cap = cv2.VideoCapture(tar_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
    cnt = startfr
    endfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) if endfr is None else endfr
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), vfps, size)
    print('Start preprocessing...')
    while cap.isOpened():
        if cnt == endfr:
            break
        print("%04d/%04d" % (cnt, endfr-1))
        cnt += 1
        _, frame = cap.read()
        writer.write(frame)
    print('Done')
    
def util2(tar_path, save_path, startfr=777, endfr=779):
    cap = cv2.VideoCapture(tar_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
    cnt = startfr
    endfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) if endfr is None else endfr
    print('Start preprocessing...')
    while cap.isOpened():
        if cnt == endfr:
            break
        print("%04d/%04d" % (cnt, endfr-1))
        cnt += 1
        _, frame = cap.read()
        cv2.imwrite('%s%04d.png' % (save_path, cnt), frame)
    print('Done')
    
if __name__ == '__main__':
    tar_path = 'target/target001.mp4'
    save_path= 'tmp/'
    util2(tar_path, save_path)