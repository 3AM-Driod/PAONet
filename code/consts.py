# -*- coding: utf-8 -*-
"""PAONet 公共常量：网格、角向求积、HG 相函数。

不依赖 DEEPRTE_pre 或任何参考求解器；VC/WQ 已固化在 grid.npz。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NS = 41
KQ = 24
H = 1.0 / 40.0

_d = np.load(Path(__file__).with_name('grid.npz'))
VC = np.asarray(_d['VC'], np.float64)          # (24, 2)
WQ = np.asarray(_d['WQ'], np.float64)          # (24,), sum = 1

INNER = np.zeros((NS, NS), bool)
INNER[1:NS - 1, 1:NS - 1] = True
INNER_IDX = np.flatnonzero(INNER.reshape(-1))


def hg_kernel(g):
    """裸 HG 相函数，按 Σ_l w_l K[k,l] = 1 行归一。"""
    vn = np.linalg.norm(VC, axis=1)
    z = np.sqrt(np.maximum(1.0 - vn ** 2, 0.0))
    O = np.concatenate([VC, z[:, None]], -1)
    mu = np.clip(O @ O.T, -1.0, 1.0)
    p = (1.0 - g ** 2) / (1.0 + g ** 2 - 2.0 * g * mu) ** 1.5
    return p / (p * WQ[None, :]).sum(1, keepdims=True)
