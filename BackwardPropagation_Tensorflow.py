#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/21 21:30
# @Author  : Colin
# @Site    : https://github.com/DreamsFuture
# @File    : BackwardPropagation_Tensorflow.py
# @Software: PyCharm Community Edition

import tensorflow as tf

I, H, O = 3, 2, 1
_input = tf.placeholder(tf.float32, [1, I])
_target = tf.placeholder(tf.float32, [1, O])
W_1 = tf.Variable(tf.zeros([3, 2]))
W_2 = tf.Variable(tf.zeros([2, 1]))

_hidden = tf.sigmoid(tf.matmul(_input, W_1))
_output = tf.sigmoid(tf.matmul(_hidden, W_2))

_loss = tf.nn.l2_loss(_output-_target)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(_loss)

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[1], [1], [1], [0]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for x, y in zip(inputs, targets):
	x += [1.]
	x = [x]
	y = [y]
	_, loss, w1, w2 = sess.run([train_step, _loss, W_1, W_2], feed_dict={_input: x, _target: y})
	print('%.3f' % loss)