# -*- coding: utf-8 -*-
"""数据集装载：DataSet\\front 下的 boundary\\ 与 volume\\ 两类，各自 27 个介质文件。

两类数据集在磁盘上、条目表上、训练/测试划分上都彼此独立（划分下标存在各文件的 TE 里）。
Y 用内存映射打开（单文件 0.3–0.5 GB，27 个介质全载入会超出内存）。
"""
from __future__ import annotations

import os

import numpy as np

from common import DATA_ROOT, source_xy

SCENE_TEST_FRAC = 0.1          # 测试场景比例（按介质划分）
SCENE_SPLIT_SEED = 20261015


class Medium:
    """一个介质的某类基元源条目集合。"""

    def __init__(self, kind, path):
        self.kind = kind
        self.path = path
        z = np.load(path, mmap_mode='r')
        self.mid = int(os.path.basename(path).split('_')[1].split('.')[0])
        self.st, self.w0, self.g = (float(v) for v in z['P'])
        self.IDX = np.asarray(z['IDX'], np.int64)          # (N,3)
        self.Y = z['Y']                                    # (N,41,41,24) log(I+1e-12)，内存映射
        self.ID = np.asarray(z['ID'], np.float32)          # (N,41,41) 直接项
        self.PHI = np.asarray(z['PHI'], np.float32)        # (N,41,41) 参考通量
        self.TE = np.asarray(z['TE'], np.int64)            # 测试条目下标
        self.NENT = len(self.IDX)
        self.KP = self.IDX[:, 2]                           # 源方向
        self.sxy = source_xy(kind, self.IDX)               # 源位置 (N,2)
        self.med = np.array([self.st, self.st * self.w0, self.g], np.float32)
        tr = np.ones(self.NENT, bool); tr[self.TE] = False
        self.TR = np.nonzero(tr)[0]


class FrontData:
    """两类数据集联合：media 列表 + 展平的 (类, 介质, 训练条目) 采样表。"""

    def __init__(self, root=DATA_ROOT, kinds=('boundary',)):
        self.root = root
        self.kinds = list(kinds)
        self.media = {k: [] for k in self.kinds}
        for k in self.kinds:
            d = os.path.join(root, k) if os.path.isdir(os.path.join(root, k)) else root
            for f in sorted(os.listdir(d)):
                if f.endswith('.npz'):
                    self.media[k].append(Medium(k, os.path.join(d, f)))
            if not self.media[k]:
                raise SystemExit('数据集为空：%s' % d)
        # 场景级划分：按介质（场景）9:1；测试场景的**全部脉冲**只用于测试，不参与训练
        allm = [(k, m) for k in self.kinds for m in self.media[k]]
        _rng = np.random.default_rng(SCENE_SPLIT_SEED)
        _n_test = int(round(SCENE_TEST_FRAC * len(allm)))
        _tst = set(int(i) for i in _rng.choice(len(allm), _n_test, replace=False))
        for _i, (_k, _m) in enumerate(allm):
            if _i in _tst:
                _m.TE = np.arange(_m.NENT); _m.TR = np.zeros(0, np.int64)
            else:
                _m.TR = np.arange(_m.NENT); _m.TE = np.zeros(0, np.int64)
        self.scene_test = np.array(sorted(_tst))
        rows = []
        for ki, k in enumerate(self.kinds):
            for mi, med in enumerate(self.media[k]):
                for a in med.TR:
                    rows.append((ki, mi, a))
        self.rows = np.asarray(rows, np.int64)
        self.gset = sorted({m.g for k in self.kinds for m in self.media[k]})
        # 每个介质在使用到的 g 列表里的下标（用于按行取用预存的散射算子）
        self.gidx = {k: [self.gset.index(m.g) for m in self.media[k]] for k in self.kinds}

    def nmedia(self, ki):
        return len(self.media[self.kinds[ki]])

    def med(self, ki, mi):
        return self.media[self.kinds[ki]][mi]

    def groups_rows(self, sel, npt):
        """把一批 (类, 介质, 条目) 按介质分组。

        sel 每行是一个 (介质, 条目) 对；训练时每个对展开成 npt 个查询节点，
        故返回的 rows 是长度 len(sel)*npt 的行掩码，直接用于切 ii/jj。
        """
        out = []
        for ki in range(len(self.kinds)):
            for mi in range(self.nmedia(ki)):
                m = (sel[:, 0] == ki) & (sel[:, 1] == mi)
                if m.any():
                    out.append((ki, mi, sel[m, 2], np.repeat(m, npt)))
        return out
