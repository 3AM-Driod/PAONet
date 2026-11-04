# -*- coding: utf-8 -*-
"""把场景 npz 与背景 npz 转换为后段训练缓存。

每个场景写入 cache/<mid>/：
    D.npy     (100,41,41,24) float16   目标对数比 Y_全场 − Y_背景
    Yb.npy    (100,41,41,24) float16   背景角向场
    Pb.npy    (100,41,41)    float32   背景积分通量
    Pf.npy    (100,41,41)    float32   完整场积分通量
    meta.npz  IDX/OCC/P_BG/P_OBJ/GEO/TE
    static.npz 由 data.scene_static() 生成的 token、查询点与介质读数

输入为已经生成好的 scenes/ 与 bg/ 数据；本脚本不调用离散求解器。
用法：
    python code/after/prep_cache.py
    MEDIA=0,20 python code/after/prep_cache.py
    FORCE=1 python code/after/prep_cache.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from common import B_DIR, CACHE, S_DIR, load_pair, scene_ids  # noqa: E402
from data import Medium, scene_static                        # noqa: E402


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
FORCE = os.environ.get('FORCE', '0') == '1'


def one(mid):
    d = os.path.join(CACHE, '%04d' % mid)
    if os.path.exists(os.path.join(d, 'D.npy')) and not FORCE:
        return 0.0
    os.makedirs(d, exist_ok=True)
    a, b = load_pair(mid)
    t0 = time.time()

    D = a['Y'].astype(np.float32) - b['Y'].astype(np.float32)
    np.save(os.path.join(d, 'D.npy'), D.astype(np.float16))
    np.save(os.path.join(d, 'Yb.npy'), b['Y'].astype(np.float16))
    np.save(os.path.join(d, 'Pb.npy'), b['PHI'].astype(np.float32))
    np.save(os.path.join(d, 'Pf.npy'), a['PHI'].astype(np.float32))

    meta = dict(IDX=a['IDX'], OCC=a['OCC'], P_BG=a['P_BG'],
                P_OBJ=a['P_OBJ'], GEO=a['GEO'],
                TE=a['TE'] if 'TE' in a.files else np.zeros(0))
    np.savez(os.path.join(d, 'meta.npz'), **meta)

    med = Medium.from_file(a)
    np.savez(os.path.join(d, 'static.npz'), **scene_static(med, mid))
    return time.time() - t0


def main():
    os.makedirs(CACHE, exist_ok=True)
    print('cache: %s' % CACHE, flush=True)
    print('scenes: %s' % S_DIR, flush=True)
    print('bg: %s' % B_DIR, flush=True)
    print('scenes to process: %d' % len(MIDS), flush=True)
    t0 = time.time()
    bad = []
    for k, mid in enumerate(MIDS):
        try:
            dt = one(mid)
        except Exception as exc:
            bad.append(mid)
            print('  skip %d: %s' % (mid, exc), flush=True)
            dt = 0.0
        if (k + 1) % 20 == 0 or k == 0:
            print('  %d/%d (%.1f s, total %.1f min)'
                  % (k + 1, len(MIDS), dt, (time.time() - t0) / 60.0), flush=True)
    if bad:
        print('skipped %d: %s%s' % (len(bad), bad[:20],
                                     ' ...' if len(bad) > 20 else ''), flush=True)
    print('done, %.1f min' % ((time.time() - t0) / 60.0), flush=True)


if __name__ == '__main__':
    main()
