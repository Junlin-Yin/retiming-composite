# -*- coding: utf-8 -*-

import cv2
from __init__ import vfps, size, Square
from retiming import retiming
from composite import composite
from visual import combine

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
        
def run_retiming():
    mp3_path = 'input/test036.mp3'
    tar_path = 'target/target001.mp4'
    tmp_path = 'reference/retiming_t36i1.npz'
    save_path = 'target/rt_target001.mp4'
    save_path, startfr = retiming(mp3_path, tar_path, tmp_path, save_path)
    print('Start from frame %04d.' % startfr)
    
def run_composite():
    sq = Square(0.25, 0.75, 0.6, 1.0)
    mp3_path = 'input/test036.mp3'
    inp_path = 'input/i36t1.npy'
    tar_path = 'target/rt_target001.mp4'
    avi_path = 'output/i36t1.avi'
    composite(inp_path, tar_path, avi_path, sq)
    outp_path = combine(avi_path, mp3_path)
    print('Final results saved in path %s.' % outp_path)
    
if __name__ == '__main__':
    run_composite()