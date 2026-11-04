# -*- coding: utf-8 -*-
"""DataLoader 化：把 slim 样本（小量 numpy）交给多进程 worker 预取，GPU 只做特征/偏置与前后向。

要点
    · 数据集项 = 一个 (场景, 脉冲) 的 slim 字典（全为定形 numpy 数组，默认 collate 即可堆叠）
    · worker 各自持有一份 AfterData（内部用记忆映射缓存，会按需切片，内存开销小）
    · pin_memory + persistent_workers + prefetch_factor：把数据准备时间藏到 GPU 计算之后
    · 计算精度保持 FP32（不做 FP16 削减）
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import NTOK
from data import AfterData

SLIM_KEYS = ('tokP', 'tokR', 'qryP', 'qryR', 'r0', 'om_raw', 'omu', 's_tab', 'tau_tab',
             'ell', 'st_bg', 'g', 'D', 'Yb', 'Phi')


class AfterDataset(Dataset):
    """按脉冲索引的数据集（场景级划分由传入的 mids 决定）。"""

    def __init__(self, mids, ntok=NTOK, cache=8):
        self.data = AfterData(ntok=ntok, verbose=False, cache=cache)
        self.index = []
        for mid in mids:
            for i in range(self.data.n_pulse(mid)):
                self.index.append((int(mid), int(i)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        mid, i = self.index[k]
        s = self.data.sample(mid, i, slim=True)
        return {key: s[key] for key in SLIM_KEYS}


def make_loader(mids, batch, nworkers=6, shuffle=True, seed=0, cache=8, drop_last=True):
    ds = AfterDataset(mids, cache=cache)
    gen = torch.Generator()
    gen.manual_seed(seed)
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=nworkers,
              pin_memory=True, drop_last=drop_last, generator=gen)
    if nworkers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)
