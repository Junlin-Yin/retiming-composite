# -*- coding: utf-8 -*-

import cv2
from __init__ import vfps, size
from retiming import retiming

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
    
def run():
    mp3_path = 'input/test036.mp3'
    tar_path = 'target/target001.mp4'
    tmp_path = 'reference/retiming_t36i1.npz'
    save_path = 'target/rt_target001.mp4'
    save_path, startfr = retiming(mp3_path, tar_path, tmp_path, save_path)
    print('Start from frame %04d.' % startfr)
    
if __name__ == '__main__':
    run()