# -*- coding:utf-8 -*-

#读取yuv并将其转化为rgb

import cv2
import numpy as np
from PIL import Image
import PIL

#读取YUV
def yuv_import(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    blk_size = np.prod(dims) * 3 // 2  #prod 数据乘法 溢出时不会出错
    fp.seek(blk_size * startfrm, 0) #seek 用于移动文件读取指针到指定位置  从0开始 便宜startfrm个位置
    Y, U, V = [], [], []
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    Yt = np.zeros((dims[0], dims[1]), np.uint8, 'C')
    Ut = np.zeros((d00, d01), np.uint8, 'C')
    Vt = np.zeros((d00, d01), np.uint8, 'C')

    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                #每次读取一个字符
                Yt[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                #每次读取一个字符
                Ut[m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                #每次读取一个字符
                Vt[m, n] = ord(fp.read(1))
        a = Yt.reshape(-1)
        #print(a.shape)
        img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
        img_final = img.reshape((288, 352, 3//2))
        #bgr_img =
        print(img_final.shape)
        print(img_final)
        fp.close()

        #return (Y, U, V)

def yuv2rgb(filename, height, width, startfrm):
    fp = open(filename, 'rb')
    framesize = height * width * 3 // 2  # 一帧图像所含的像素个数
    h_h = height // 2
    h_w = width // 2

    fp.seek(0, 2)  # 设置文件指针到文件流的尾部
    ps = fp.tell()  # 当前文件指针位置
    numfrm = ps // framesize  # 计算输出帧数
    fp.seek(framesize * startfrm, 0)

    for i in range(numfrm - startfrm):
        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')

        for m in range(height):
            for n in range(width):
                Yt[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                Ut[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                Vt[m, n] = ord(fp.read(1))

        img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
        img = img.reshape((height * 3 // 2, width)).astype('uint8')

        #yuv2rgb
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
        #print(bgr_img)
        cv2.namedWindow('show', 0)
        cv2.imshow('show', bgr_img)
       # cv2.waitKey(10)


        img = cv2.resize(bgr_img, (688, 560), interpolation=cv2.INTER_AREA)
        cv2.imwrite('result/yuv/ori/' + str(i) + '.jpg', img)
        #cv2.imwrite('result/007.jpg', bgr_img)
        #cv2.imwrite('yuv2bgr/%d.jpg' % (i + 1), bgr_img)
        #print("Extract frame %d " % (i + 1))

if __name__=='__main__':
    width = 352
    height = 288
    #yuv_import('result/output.yuv', (height, width), 1, 0)
    #print(data)
    #YY = data[0][0]
    #print(YY)
    # print("图片转换中……")
    # print(YY)
    # cv2.namedWindow('show', 0)
    # cv2.imshow("show", data)
    # print("图片显示成功")

    #im = Image.frombytes('L', (352, 288, 3), YY)

    yuv2rgb('result/foreman.yuv', height, width, 0)

    #im.show()
    #cv2.waitKey(0)
    
