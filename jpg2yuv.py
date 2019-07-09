# -*- coding:utf-8 -*-

import cv2
import PIL
from PIL import Image
from ffmpy3 import FFmpeg
import os

in_jpgDatasetPath = 'result/yuv/jpg'
out_yuvDatasetPath = 'result/yuv'

piclist = os.listdir(in_jpgDatasetPath) #获取图片列表

for pic in piclist:
    picname = pic.split('.')[0]
    picpath = os.path.join(in_jpgDatasetPath, pic)
    img = Image.open(picpath)
    in_wid, in_hei = img.size

    out_wid = in_wid // 16 * 16
    out_hei = in_hei // 16 * 16
    size = '{}x{}'.format(out_wid, out_hei)  # 输出文件会缩放成这个大小

    outname = out_yuvDatasetPath + '/' + picname + '_' + size + '.yuv'
    ff = FFmpeg(inputs={picpath:None}, outputs={outname:'-s {} -pix_fmt yuv420p'.format(size)})
    print(ff.cmd)
    ff.run()
