# -*- coding: utf-8 -*-
"""前段网络公共定义：网格、角向求积、散射算子、路径。

相函数定义：μ 取三维嵌入 O = [VC, sqrt(1-|VC|²)] 的单位向量点积，
p ∝ (1-g²)/(1+g²-2gμ)^1.5，返回**裸相函数**并按加权行和归一 Σ_l w_l K = 1。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import consts as C                       # noqa: E402
from config import FRONT_DATA, WEIGHTS   # noqa: E402

NS = C.NS                            # 41 节点
KQ = C.KQ                            # 24 方向
H = C.H                              # 1/40
VC = np.asarray(C.VC, np.float64)    # (24,2) 方向表
WQ = np.asarray(C.WQ, np.float64)    # (24,) 求积权重，Σ = 1
INNER = np.zeros((NS, NS), bool); INNER[1:NS - 1, 1:NS - 1] = True   # 内区 = 排除最边缘一行/列

DATA_ROOT = os.environ.get('DATA_ROOT', str(FRONT_DATA))
WEIGHT_DIR = str(WEIGHTS / 'runs' / 'front')


def new_run_dir(root=WEIGHT_DIR):
    """自动创建 train1、train2、… 子目录以区分每次训练；空目录（上次中断）会被复用。"""
    os.makedirs(root, exist_ok=True)
    n = 1
    while True:
        d = os.path.join(root, 'train%d' % n)
        if not os.path.exists(os.path.join(d, 'front_net.pt')):
            os.makedirs(d, exist_ok=True)
            return d
        n += 1


def hg_kernel(g):
    """裸相函数，Σ_l w_l K = 1。"""
    return np.asarray(C.hg_kernel(g), np.float64)


def pk_of(g):
    """散射算子 PK[i,j] = ω_j·K[i,j]（行和 = 1），网络散射块直接用它做角向求和。"""
    return (hg_kernel(g) * WQ[None, :]).astype(np.float32)


def source_xy(kind, idx):
    """条目源位置：边界条目 (侧, 节点, 方向) -> 墙上坐标；体条目 (i, j, 方向) -> 内部节点坐标。"""
    idx = np.asarray(idx, np.int64)
    out = np.empty((len(idx), 2), np.float64)
    if kind == 'boundary':
        s, p = idx[:, 0], idx[:, 1] / float(NS - 1)
        for side in range(4):
            m = s == side
            if side == 0:
                out[m] = np.stack([np.zeros(m.sum()), p[m]], -1)
            elif side == 1:
                out[m] = np.stack([np.ones(m.sum()), p[m]], -1)
            elif side == 2:
                out[m] = np.stack([p[m], np.zeros(m.sum())], -1)
            else:
                out[m] = np.stack([p[m], np.ones(m.sum())], -1)
    else:
        out[:, 0] = idx[:, 0] / float(NS - 1)
        out[:, 1] = idx[:, 1] / float(NS - 1)
    return out
