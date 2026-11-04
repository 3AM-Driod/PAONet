# -*- coding: utf-8 -*-
"""Per-scene evaluation of a front-net checkpoint.

A scene's entries are summed first, then compared with the summed reference
inner field. Reads ZERO_COLS / ZERO_ENC / NO_IDIR / OUT_LIN / MODE_FRONT.

Usage:
    python code/front/eval_front_scene.py --ckpt weights/formal/front/front_net.pt \
        --data data/front/data1 --kind boundary --tag formal_front_scene
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)

from config import RESULTS                    # noqa: E402
from common import INNER                      # noqa: E402
from data import FrontData                    # noqa: E402
from front_net import DEV, FrontNet, MODE_FRONT, OUT_LIN, phi_of_entries   # noqa: E402


def rel(a, b):
    a = np.asarray(a, np.float64).reshape(-1)[INNER.ravel()]
    b = np.asarray(b, np.float64).reshape(-1)[INNER.ravel()]
    return 100.0 * float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--kind', default='boundary')
    ap.add_argument('--tag', default='scene')
    a = ap.parse_args()

    os.environ['DATA_ROOT'] = a.data
    data = FrontData(root=a.data, kinds=(a.kind,))
    net = FrontNet().to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    net.eval()

    errs = []
    with torch.no_grad():
        for mi in range(data.nmedia(0)):
            med = data.med(0, mi)
            if len(med.TE) == 0:
                continue
            Phi = phi_of_entries(net, med, med.TE)
            ref = np.asarray(med.PHI)[med.TE]
            errs.append(rel(Phi.sum(0), ref.sum(0)))

    e = np.asarray(errs)
    out = dict(tag=a.tag, n_scene=int(e.size), mean=float(e.mean()),
               median=float(np.median(e)), p90=float(np.percentile(e, 90)),
               p99=float(np.percentile(e, 99)), max=float(e.max()),
               mode_front=MODE_FRONT, out_lin=bool(OUT_LIN), ckpt=a.ckpt)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(str(RESULTS), exist_ok=True)
    with open(os.path.join(str(RESULTS), 'eval_front_scene_%s.json' % a.tag), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
