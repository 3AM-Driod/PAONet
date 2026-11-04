# -*- coding: utf-8 -*-
"""PAONet 路径配置。所有代码默认从本文件推导路径，不依赖 E:\\RTE_JEPA\\RTE_DL。"""
from __future__ import annotations

import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent      # .../PAONet
CODE = ROOT / 'code'
DATA = ROOT / 'data'
WEIGHTS = ROOT / 'weights'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

FRONT_DATA = DATA / 'front' / 'data1'
AFTER_SCENES = DATA / 'after' / 'data1' / 'scenes'
AFTER_BG = DATA / 'after' / 'data1' / 'bg'
AFTER_CACHE = DATA / 'after' / 'cache1'
