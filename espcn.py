# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import sys
import pdb

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

class ESPCN:
    def __init__(self, filters_size, channels, ratio, batch_size, lr_size, edge):
        self.filters_size = filters_size
        self.channels = channels
        self.ratio = ratio
        self.batch_size = batch_size
        self.lr_size = lr_size
        self.edge = edge
        self.variables = self.create_variables()

    def create_variables(self):
        var = dict()
        var['filters'] = list()
        # the input layer
        #第一层   5 * 5   64
        var['filters'].append(
            create_variable('filter',
                            [self.filters_size[0],
                             self.filters_size[0],
                             1,                           #这不应该是3么
                             self.channels[0]]))
        # the hidden layers
        #只有一层 由1循环到2  3 * 3 * 32
        for idx in range(1, len(self.filters_size) - 1):
            var['filters'].append(
                create_variable('filter', 
                                [self.filters_size[idx],
                                 self.filters_size[idx],
                                 self.channels[idx - 1],
                                 self.channels[idx]]))
        # the output layer
        #pixel shuffle 3 * 3  输出为结果为 h * w * r^2    为啥不是 * 3 * r^2?(r^2是卷积核的输出， 3在图片中)
        var['filters'].append(
            create_variable('filter',
                            [self.filters_size[-1],
                             self.filters_size[-1],
                             self.channels[-1],
                             self.ratio**2]))

        var['biases'] = list()
        for channel in self.channels:
            var['biases'].append(create_bias_variable('bias', [channel]))
        var['biases'].append(create_bias_variable('bias', [self.ratio**2]))


        image_shape = (self.batch_size, self.lr_size, self.lr_size, 3)
        var['images'] = tf.placeholder(tf.uint8, shape=image_shape, name='images')

        #为什么要  - edge 还有为什么lr 和  hr 一样大
        label_shape = (self.batch_size, self.lr_size - self.edge, self.lr_size - self.edge, 3 * self.ratio**2)
        var['labels'] = tf.placeholder(tf.uint8, shape=label_shape, name='labels')

        return var

    def build_model(self):
        #由create方法得来
        images, labels = self.variables['images'], self.variables['labels']
        input_images, input_labels = self.preprocess([images, labels])
        output = self.create_network(input_images)
        reduced_loss = self.loss(output, input_labels)
        return reduced_loss, images, labels
    #
    def save(self, sess, saver, logdir, step):
        # print('[*] Storing checkpoint to {} ...'.format(logdir), end="")
        sys.stdout.flush()  #刷新缓冲区 实时输出显示

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        checkpoint = os.path.join(logdir, "model.ckpt")
        saver.save(sess, checkpoint, global_step=step)
        # print('[*] Done saving checkpoint.')

    def load(self, sess, saver, logdir):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logdir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(logdir, ckpt_name))
            return True
        else:
            return False

    def preprocess(self, input_data):
        # cast to float32 and normalize the data
        #input_data : [iamges, labels]
        input_list = list()
        for ele in input_data:
            if ele is None:
                continue
            ele = tf.cast(ele, tf.float32) / 255.0
            print(ele)
            input_list.append(ele)

        print("------------------------------")
        print(input_list)
        print("------------------------------")
        #input_list 含有image 和 label
        #最后一维 应该是通道 为什么不是3呢
        input_images, input_labels = input_list[0][:,:,:,0:1], None
        # Generate doesn't use input_labels
        ratioSquare = self.ratio * self.ratio
        if input_data[1] is not None:
            #label 是r^2通道
            input_labels = input_list[1][:,:,:,0:ratioSquare]
        return input_images, input_labels

    def create_network(self, input_image):
        '''The default structure of the network is:

        input (3 channels) ---> 5 * 5 conv (64 channels) ---> 3 * 3 conv (32 channels) ---> 3 * 3 conv (3*r^2 channels)

        Where `conv` is 2d convolutions with a non-linear activation (tanh) at the output.
        '''
        current_layer = input_image

        for idx in range(len(self.filters_size)):
            conv = tf.nn.conv2d(current_layer, self.variables['filters'][idx], [1, 1, 1, 1], padding='VALID')
            with_bias = tf.nn.bias_add(conv, self.variables['biases'][idx])
            if idx == len(self.filters_size) - 1:  #最后一层 没有激活
                current_layer = with_bias
            else:
                current_layer = tf.nn.tanh(with_bias)
        return current_layer

    def loss(self, output, input_labels):
        residual = output - input_labels
        loss = tf.square(residual)
        reduced_loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', reduced_loss)
        return reduced_loss

    def generate(self, lr_image):
        lr_image = self.preprocess([lr_image, None])[0]
        sr_image = self.create_network(lr_image)
        sr_image = sr_image * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)
        return sr_image
