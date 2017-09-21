#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/21 21:29
# @Author  : Colin
# @Site    : https://github.com/DreamsFuture
# @File    : BackwardPropagation_Python.py
# @Software: PyCharm Community Edition

# coding=utf-8
# !/bin/python
import sys
import os
import math


# 请完成下面的函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************
def sigmoid(inputs):
    return [1.0 / (1.0 + math.exp(-x)) for x in inputs]


# sigmoid 的导数
def dsigmoid(outputs, grad_outputs):
    return [x * (1.0 - x) * y for x, y in zip(outputs, grad_outputs)]


# 损失函数
def loss(y, label):
    return sum((a - b) ** 2 for a, b in zip(y, label)) / 2.


# 损失函数的导数
def dloss(y, label):
    return [a - b for a, b in zip(y, label)]


# 矩阵乘法
def matmul(inputs, M, n, m):
    output = [0.0] * m
    for i in range(n):
        for j in range(m):
            output[j] += inputs[i] * M[i][j]
    return output


# 矩阵乘法导数
def dmatmul(inputs, M, n, m, grad_outputs):
    grad_M = [[0.0] * m for _ in range(n)]
    grad_input = [0.0] * n
    for i in range(n):
        for j in range(m):
            grad_M[i][j] += grad_outputs[j] * inputs[i]
            grad_input[i] += grad_outputs[j] * M[i][j]
    return grad_input, grad_M


# 更新矩阵
def update(M, grad_M, n, m):
    for i in range(n):
        for j in range(m):
            M[i][j] -= 0.5 * grad_M[i][j]


def bpnn(N, I, H, O, inputs, targets):
    I += 1
    W_1 = [[0.0] * H for _ in range(I)]
    W_2 = [[0.0] * O for _ in range(H)]

    res = []
    for _input, _target in zip(inputs, targets):
        _input += [1.0]
        _hidden = matmul(_input, W_1, I, H)
        _actived_hidden = sigmoid(_hidden)
        _output = matmul(_actived_hidden, W_2, H, O)
        _actived_ouput = sigmoid(_output)
        _loss = loss(_actived_ouput, _target)

        _grad_actived_output = dloss(_actived_ouput, _target)
        _grad_output = dsigmoid(_actived_ouput, _grad_actived_output)
        _grad_actived_hidden, _grad_W_2 = dmatmul(_actived_hidden, W_2, H, O, _grad_output)
        _grad_hidden = dsigmoid(_actived_hidden, _grad_actived_hidden)
        _, _grad_W_1 = dmatmul(_input, W_1, I, H, _grad_hidden)

        update(W_1, _grad_W_1, I, H)
        update(W_2, _grad_W_2, H, O)
        res.append(_loss)

    return res


# ******************************结束写代码******************************


_input = raw_input()

_N, _I, _H, _O = _input.split()
_N = int(_N)
_I = int(_I)
_H = int(_H)
_O = int(_O)

_inputs = []
_targets = []
for _inputs_i in xrange(_N):
    _inputs_temp = map(int, raw_input().strip().split(' '))
    _inputs.append(_inputs_temp)
    _targets_temp = map(int, raw_input().strip().split(' '))
    _targets.append(_targets_temp)

res = bpnn(_N, _I, _H, _O, _inputs, _targets)

for res_cur in res:
    print( "%.3f" % res_cur)

