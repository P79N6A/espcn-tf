import argparse
import tensorflow as tf
from scipy import ndimage
from scipy import misc
import numpy as np
from prepare_data import *
from psnr import psnr
import json
import pdb
import datetime
import time
from utils import yuv_import, yuv2rgb
import cv2

from espcn import ESPCN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_arguments():
    parser = argparse.ArgumentParser(description='EspcnNet generation script')
    parser.add_argument('--checkpoint', type=str,default="logdir_2x/0703/train",
                        help='Which model checkpoint to generate from')
    parser.add_argument('--lr_image', type=str,default="result/foreman.yuv",
                        help='The low-resolution image waiting for processed.')
    parser.add_argument('--hr_image', type=str,
                        help='The high-resolution image which is used to calculate PSNR.')
    parser.add_argument('--out_path', type=str,default="result/yuv/007_0703",
                        help='The output p  ath for the super-resolution image')
    return parser.parse_args()

def check_params(args, params):
    if len(params['filters_size']) - len(params['channels']) != 1:
        print("The length of 'filters_size' must be greater then the length of 'channels' by 1.")
        return False
    return True

def generate():
    args = get_arguments()

    with open("./params2.json", 'r') as f:
        params = json.load(f)

    if check_params(args, params) == False:
        return

    sess = tf.Session()


    net = ESPCN(filters_size=params['filters_size'],
                   channels=params['channels'],
                   ratio=params['ratio'],
                   batch_size=1,
                   lr_size=params['lr_size'],
                   edge=params['edge'])

    loss, images, labels = net.build_model()


    #rgb
    lr_image = tf.placeholder(tf.uint8)

    saver = tf.train.Saver()
    try:
        model_loaded = net.load(sess, saver, args.checkpoint)
    except:
        raise Exception(
            "Failed to load model, does the ratio in params2.json match the ratio you trained your checkpoint with?")

    if model_loaded:
        print("[*] Checkpoint load success!")
    else:
        print("[*] Checkpoint load failed/no checkpoint found")
        return

    def run_net(num):
        start0 = time.time()
        start1 = time.time()
        sr_image_y_data = sess.run(sr_image, feed_dict={lr_image: lr_image_batch})
        print('1', time.time() - start1)   #0.1967

        start2 = time.time()
        # pixel shuffle  b c r^2 h w ---> b c rh rw
        sr_image_y_data = shuffle(sr_image_y_data[0], params['ratio'])
        print('2', time.time() - start2)  #0.2775
        start3 = time.time()
        sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                            params['ratio'] * np.array(lr_image_data.shape[0:2]),
                                            'bicubic')
        edge = params['edge'] * params['ratio'] // 2

        sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge, edge:-edge, 1:3]),
                                             axis=2)
        print('3', time.time() - start3) #0.0219
        start4 = time.time()
        #sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)
        print('4', time.time() - start4)#3.7009   86.59%

        # start5 = time.time()
        # # res_image = cv2.cvtColor(sr_image_data, cv2.COLOR_BGR2YUV)
        print(type(sr_image_ycbcr_data))
        fw = open("result/res.yuv", 'ab')
        fw.write(sr_image_ycbcr_data)
        end = time.time()

        #print(sr_image_data.shape)
        # cv2.namedWindow('show_sr', 0)
        # cv2.imshow('show_sr', sr_image_ycbcr_data)
        # cv2.waitKey(50)
        # #print('5', time.time() - start5) #0.0767
        # cv2.imwrite(args.out_path + '_' + str(num) + '.jpg', sr_image_ycbcr_data)
        #misc.imsave(args.out_path + '_' + str(num) + '.png', sr_image_data)
        print("{:f} seconds".format(time.time() - start0))  #4.2739

    if args.lr_image[-3:] == 'yuv':
        width = 352
        height = 288
        # #lr_image_yuv_data = data[0][0]
        # #lr_image_yuv_data = misc.imread(args.lr_image)
        # #print(type(args.lr_image))
        # lr_image_yuv_data = yuv_import(args.lr_image, (height, width), 1, 0)
        # print(lr_image_yuv_data)
        # lr_image_y_data = lr_image_yuv_data
        # #print(lr_image_y_data.shape)
        # # lr_image_cb_data = lr_image_yuv_data[:, :, 1:2]
        # # lr_image_cr_data = lr_image_yuv_data[:, :, 2:3]
        # lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        # lr_image_batch[0] = lr_image_y_data
        fp = open(args.lr_image, 'rb')
        framesize = height * width * 3 // 2  # 一帧图像所含的像素个数
        h_h = height // 2
        h_w = width // 2

        fp.seek(0, 2)  # 设置文件指针到文件流的尾部
        ps = fp.tell()  # 当前文件指针位置
        numfrm = ps // framesize  # 计算输出帧数
        fp.seek(framesize * 0, 0)

        for i in range(10 - 0):
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

            # yuv2rgb
            bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
            # print(bgr_img)
            cv2.namedWindow('show', 0)
            cv2.imshow('show', bgr_img)
            #cv2.waitKey(10)
            # cv2.imwrite('result/007.jpg', bgr_img)
            # cv2.imwrite('yuv2bgr/%d.jpg' % (i + 1), bgr_img)
            # print("Extract frame %d " % (i + 1))
            lr_image_data = bgr_img
            # print(lr_image_data)
            lr_image_data = lr_image_data.reshape(lr_image_data.shape[0], lr_image_data.shape[1], 3, )
            print(lr_image_data.shape)
            lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
            lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
            print('************************')
            print(lr_image_y_data.shape)
            lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
            lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
            lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
            lr_image_batch[0] = lr_image_y_data

            sr_image = net.generate(lr_image)
            run_net(i)

    else:
        #imghdr.what(args.lr_image) == 'jpeg' or imghdr.what(args.lr_image) == 'png' or imghdr.what(args.lr_image) == 'bmp':
        lr_image_data = misc.imread(args.lr_image)
        #print(lr_image_data)
        lr_image_data = lr_image_data.reshape( lr_image_data.shape[0], lr_image_data.shape[1],3,)
        print(lr_image_data.shape)
        lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
        print('************************')
        print(lr_image_y_data.shape)
        lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
        lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        lr_image_batch[0] = lr_image_y_data
        sr_image = net.generate(lr_image)

        #run_net()


    # if args.hr_image != None:
    #     hr_image_data = misc.imread(args.hr_image)
    #     model_psnr = psnr(hr_image_data, sr_image_data, edge)
    #     print('PSNR of the model: {:.2f}dB'.format(model_psnr))
    #
    #     sr_image_bicubic_data = misc.imresize(lr_image_data,
    #                                     params['ratio'] * np.array(lr_image_data.shape[0:2]),
    #                                     'bicubic')
    #     misc.imsave(args.out_path + '_bicubic.png', sr_image_bicubic_data)
    #     bicubic_psnr = psnr(hr_image_data, sr_image_bicubic_data, 0)
    #     print('PSNR of Bicubic: {:.2f}dB'.format(bicubic_psnr))




if __name__ == '__main__':
    generate()