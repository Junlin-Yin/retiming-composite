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