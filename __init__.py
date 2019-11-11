# -*- coding: utf-8 -*-

import dlib
pdctdir = 'reference/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pdctdir)

tar_dir = 'target/'
inp_dir = 'input/'
ref_dir = 'reference/'
outp_dir= 'output/'

vfps = 30
size = (1280, 720)

class Square:
    def __init__(self, l, r, u, d):
        self.left = l
        self.right = r
        self.up = u
        self.down = d
        
    def align(self, S):
        from math import ceil, floor
        left  = ceil(self.left  * S)
        right = floor(self.right * S)
        upper = ceil(self.up    * S)
        lower = floor(self.down  * S)
        return left, right, upper, lower
    
    def getrsize(self, sp):
        rs1 = sp[1] / (self.right - self.left)
        rs2 = sp[0] / (self.down - self.up)
        return round((rs1+rs2)/2)