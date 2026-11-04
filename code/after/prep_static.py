# -*- coding: utf-8 -*-
"""只重算 cache/<mid>/static.npz，不改动 D/Yb/Pb/Pf。

static.npz 包含：
    tokP / tokR / qryP / qryR
由 data.scene_static() 依据当前连续界面折线采样器生成。

输入为已有 cache/<mid>/meta.npz；本脚本不调用离散求解器。
用法：
    python code/after/prep_static.py
    MEDIA=0,20 python code/after/prep_static.py
    FORCE=1 python code/after/prep_static.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from common import CACHE, scene_ids          # noqa: E402
from data import Medium, scene_static        # noqa: E402


def _media_ids(raw):
    if not raw.strip():
        return scene_ids()
    v = [int(x) for x in raw.split(',') if x.strip()]
    if len(v) == 1:
        return v
    if len(v) == 2:
        return list(range(v[0], v[1]))
    return v


MIDS = _media_ids(os.environ.get('MEDIA', ''))
FORCE = os.environ.get('FORCE', '1') == '1'


def one(mid):
    d = os.path.join(CACHE, '%04d' % mid)
    mp = os.path.join(d, 'meta.npz')
    if not os.path.exists(mp):
        return 0.0, False
    if os.path.exists(os.path.join(d, 'static.npz')) and not FORCE:
        return 0.0, True
    a = np.load(mp, allow_pickle=False)
    med = Medium.from_file(a)
    np.savez(os.path.join(d, 'static.npz'), **scene_static(med, mid))
    return 1.0, True


def main():
    print('cache: %s' % CACHE, flush=True)
    print('scenes to process: %d' % len(MIDS), flush=True)
    t0 = time.time()
    ok, miss = 0, []
    for k, mid in enumerate(MIDS):
        dt, found = one(mid)
        if found:
            ok += 1
        else:
            miss.append(mid)
        if (k + 1) % 20 == 0 or k == 0:
            print('  %d/%d (total %.1f min)'
                  % (k + 1, len(MIDS), (time.time() - t0) / 60.0), flush=True)
    if miss:
        print('missing meta: %s%s' % (miss[:20], ' ...' if len(miss) > 20 else ''),
              flush=True)
    print('done: %d scenes, %.1f min' % (ok, (time.time() - t0) / 60.0), flush=True)


if __name__ == '__main__':
    main()
