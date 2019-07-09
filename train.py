# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import json
import time

import tensorflow as tf
from reader import create_inputs
from espcn import ESPCN

import pdb

BATCH_SIZE = 64
NUM_EPOCHS = 5000
LEARNING_RATE = 0.001
LOGDIR_ROOT = './logdir_{}x'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_arguments():
    parser = argparse.ArgumentParser(description='EspcnNet example network')
    parser.add_argument('--checkpoint', type=str,
                        help='Which model checkpoint to load from', default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many image files to process at once.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--decay_rate', type=float, default=0.96,
                        help='Learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=10000,
                        help='Learning rate decay steps')

    return parser.parse_args()

def check_params(args, params):
    if len(params['filters_size']) - len(params['channels']) != 1:  #前两个channel为手工决定 最后一个由r决定
        print("The length of 'filters_size' must be greater then the length of 'channels' by 1.")
        return False
    return True

def train():
    args = get_arguments()

    with open("./params2.json", 'r') as f:
        params = json.load(f)
        #print(params)

    if check_params(args, params) == False:
        return

    logdir_root = args.logdir_root # ./logdir
    if logdir_root == LOGDIR_ROOT:
        logdir_root = logdir_root.format(params['ratio']) # ./logdir_{RATIO}x
    logdir = os.path.join(logdir_root, 'train') # ./logdir_{RATIO}x/train

    # Load training data as np arrays
    lr_images, hr_labels = create_inputs(params)  #读取lr hr

    #init net
    net = ESPCN(filters_size=params['filters_size'],
                   channels=params['channels'],
                   ratio=params['ratio'],
                   batch_size=args.batch_size,
                   lr_size=params['lr_size'],
                   edge=params['edge'])

    loss, images, labels = net.build_model()

    #end_learning_rate =
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_step, args.decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # set up logging for tensorboard
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()

    # set up session
    sess = tf.Session()

    # saver for storing/restoring checkpoints of the model
    saver = tf.train.Saver(max_to_keep=0)

    init = tf.global_variables_initializer()
    sess.run(init)

    if net.load(sess, saver, logdir):
        print("[*] Checkpoint load success!")
    else:
        print("[*] Checkpoint load failed/no checkpoint found")


    #start training
    try:
        steps, start_average, end_average = 0, 0, 0
        start_time = time.time()
        for ep in range(1, args.epochs + 1):
            batch_idxs = len(lr_images) // args.batch_size
            batch_average = 0
            for idx in range(0, batch_idxs):
                # On the fly batch generation instead of Queue to optimize GPU usage
                start_batch_time = time.time()
                batch_images = lr_images[idx * args.batch_size : (idx + 1) * args.batch_size]
                batch_labels = hr_labels[idx * args.batch_size : (idx + 1) * args.batch_size]
                
                steps += 1
                summary, loss_value, lr, _ = sess.run([summaries, learning_rate, loss, optim], feed_dict={images: batch_images, labels: batch_labels, global_step: steps})
                writer.add_summary(summary, steps)
                batch_average += loss_value

                if steps % 100 == 0:
                    print('------Epoch: {}, step: {:d}, lr:{:.6f}, loss: {:.9f}, ({:.3f} sec)'.format(ep, steps, lr, loss_value,
                                                                                           time.time() - start_batch_time))
            # Compare loss of first 20% and last 20%  ？？？？？这是干啥的 哦哦哦 用来计算最后提高了多少的
            batch_average = float(batch_average) / batch_idxs
            if ep < (args.epochs * 0.2):
                start_average += batch_average
            elif ep >= (args.epochs * 0.8):
                end_average += batch_average

            duration = time.time() - start_time
            print('Epoch: {}, step: {:d}, loss: {:.9f}, ({:.3f} sec/epoch)'.format(ep, steps, batch_average, duration))
            start_time = time.time()
            net.save(sess, saver, logdir, steps)
    except KeyboardInterrupt:
        print()
    finally:
        start_average = float(start_average) / (args.epochs * 0.2)
        end_average = float(end_average) / (args.epochs * 0.2)
        print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
          % (start_average, end_average, 100 - (100*end_average/start_average)))

if __name__ == '__main__':
    train()
