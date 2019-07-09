# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np

file_dir = 'result/yuv/ori/'
file_list = os.listdir(file_dir)

fps = 24

size = (688, 560)

video_writer = cv2.VideoWriter("ori.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#video_writer = cv2.VideoWriter("soapbox.avi", fourcc, fps, size)


file_list.sort(key=lambda x: int(x[:-4]))
for item in file_list:
    item = file_dir + item
    img = cv2.imread(item)
    video_writer.write(img)
    cv2.namedWindow('111', 0)
    cv2.imshow('111', img)
    cv2.waitKey(20)
    print('write:', item)

video_writer.release()