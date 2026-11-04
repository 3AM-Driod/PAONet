# -*- coding: utf-8 -*-
"""解析弹道项（ω0=0 的直接项）与脉冲辅助类。

只实现沿特征线的一次扫掠，不包含任何迭代散射求解，因此不依赖参考求解器。
与数据集生成时使用的 direct_multi 逐位一致（同一迎风离散）。
"""
from __future__ import annotations

import numpy as np

from consts import H, KQ, NS, VC

_PLAN_CACHE = {}


def build_plans(NX, NY, VC):
    key = (NX, NY, VC.shape[0])
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    cls = {}
    for k in range(VC.shape[0]):
        cls.setdefault((1 if VC[k, 0] > 0 else -1, 1 if VC[k, 1] > 0 else -1), []).append(k)
    plans = {}
    for (sx, sy), ks in cls.items():
        dias = []
        for n in range(1, NX + NY - 1):
            I = np.arange(max(0, n - (NY - 1)), min(n, NX - 1) + 1)
            J = n - I
            m = (I >= 1) & (J >= 1)
            if not m.any():
                continue
            I, J = I[m], J[m]

            def tr(II, JJ):
                i = II if sx > 0 else NX - 1 - II
                j = JJ if sy > 0 else NY - 1 - JJ
                return i * NY + j
            dias.append((tr(I, J), tr(I - 1, J), tr(I, J - 1)))
        plans[(sx, sy)] = (np.asarray(ks), dias)
    _PLAN_CACHE[key] = plans
    return plans


def direct_multi(NX, NY, H, VC, mut, bcw=None, injw=None, chunk=256):
    """ω0=0 的多右端项直接项：按流向一次 Gauss-Seidel 扫掠。"""
    NP, K = NX * NY, VC.shape[0]
    M = (bcw[0].shape[-1] if bcw is not None else injw.shape[-1])
    mut_f = np.asarray(mut, float).ravel()
    plans = build_plans(NX, NY, VC)
    A = np.abs(VC[:, 0]) / H
    B = np.abs(VC[:, 1]) / H
    out = np.empty((NP, K, M))
    ar = np.arange(NX)
    for a0 in range(0, M, chunk):
        a1 = min(a0 + chunk, M)
        I = np.zeros((NP, K, a1 - a0))
        if bcw is not None:
            for k in range(K):
                vx, vy = VC[k]
                for s in range(4):
                    col = bcw[s][:, k, a0:a1]
                    if not col.any():
                        continue
                    if s == 0 and vx > 0:
                        I[0:NY, k, :] = col
                    elif s == 1 and vx < 0:
                        I[(NX - 1) * NY:NX * NY, k, :] = col
                    elif s == 2 and vy > 0:
                        I[ar * NY, k, :] = col
                    elif s == 3 and vy < 0:
                        I[ar * NY + (NY - 1), k, :] = col
        SRC = injw[:, :, a0:a1] if injw is not None else None
        for (sx, sy), (ks, dias) in plans.items():
            Ak = A[ks][None, :, None]
            Bk = B[ks][None, :, None]
            den_k = (A[ks] + B[ks])[None, :, None]
            for io, i1, i2 in dias:
                den = den_k + mut_f[io][:, None, None]
                num = Ak * I[np.ix_(i1, ks)] + Bk * I[np.ix_(i2, ks)]
                if SRC is not None:
                    num = num + SRC[np.ix_(io, ks)]
                I[np.ix_(io, ks)] = num / den
        out[:, :, a0:a1] = I
    return out.reshape(NX, NY, K, M)


def source_xy(idx):
    """边界条目 (侧, 节点, 方向) -> 墙上坐标 (M,2)。"""
    idx = np.asarray(idx, np.int64)
    p = idx[:, 1] / float(NS - 1)
    out = np.empty((len(idx), 2), np.float64)
    s = idx[:, 0]
    m = s == 0; out[m] = np.stack([np.zeros(m.sum()), p[m]], -1)
    m = s == 1; out[m] = np.stack([np.ones(m.sum()), p[m]], -1)
    m = s == 2; out[m] = np.stack([p[m], np.zeros(m.sum())], -1)
    m = s == 3; out[m] = np.stack([p[m], np.ones(m.sum())], -1)
    return out


class FrontMed:
    """前段 phi_of_entries 需要的最小介质对象（直接项由解析式给出）。"""

    def __init__(self, st, w0, g, idx, ID):
        self.st, self.w0, self.g = float(st), float(w0), float(g)
        self.med = np.array([st, st * w0, g], np.float32)
        self.IDX = np.asarray(idx, np.int64)
        self.KP = self.IDX[:, 2]
        self.sxy = source_xy(self.IDX)
        self.ID = ID
        self.Y = None
        self.PHI = None
        self.TE = np.arange(len(idx))
        self.TR = np.zeros(0, np.int64)


def ballistic_ids(idx, st):
    """每个条目的离散直接项 (M,NS,NS) 与完整 (NS,NS,KQ,M) 直接场。"""
    M = len(idx)
    bcw = [np.zeros((NS, KQ, M)) for _ in range(4)]
    for m, (s, p, k) in enumerate(np.asarray(idx, np.int64)):
        bcw[s][p, k, m] = 1.0
    I0 = direct_multi(NS, NS, H, VC, np.full(NS * NS, float(st)), bcw=bcw)
    beam = np.take_along_axis(I0, np.asarray(idx, np.int64)[:, 2][None, None, None, :],
                              axis=2)[..., 0, :]
    return np.ascontiguousarray(beam.transpose(2, 0, 1)), I0
