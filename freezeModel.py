# -*- coding:utf-8 -*-

"""
 This file used to freeze tensorflow .ckpt to .pb
"""

import tensorflow as tf


#两种方式 方法1：函数方法，传入session
def freeze_session(session, keep_var_name=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        #difference方法 返回的值在global_variables中单不在keep_var_name中
        freeze_var_name = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_name or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.decive = ''

        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_name)
        return frozen_graph

#方法1第二步 将冻结的模型保存为pb格式
#其中上一个函数中 output_name保存的就是节点
session = tf.Session()
net_model = '读取网络模型'
output_path = ''
pb_model_name = 'xxxx.pb'
frozen_graph = freeze_session(session, output_names=[net_model.output.op.name])
tf.python.framework.graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)

#------------------------------------------------------------------------------------------------------------
#方法二 直接冻结
#1.指定模型输出
output_nodes = tf.global_variables()
#utput_nodes = ["Accuracy/prediction", "Metric/Dice"] 指定模型输出, 这样可以允许自动裁剪无关节点. 这里认为使用逗号分割

#加载模型
saver = tf.train.import_meta_graph('model.ckpt.meta', clear_devices=True)

with tf.Session(graph=tf.get_default_graph()) as sess:
    #序列化模型
    input_graph_def = sess.graph.as_graph_def()
    #载入权重
    saver.restore(sess, 'model.ckpt')
    #转换变量为常量
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_nodes)

    #写入pb文件
    with open('frozen_model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())

