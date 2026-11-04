# -*- coding: utf-8 -*-
"""Pulse-level evaluation of a front-net checkpoint on all test entries of data1.

Protocol matches the paper: 100 test scenes x 100 entries = 1e4 pulses,
error = 100*||(Phi_hat-Phi_ref)[inner]|| / ||Phi_ref[inner]||.
Reads MODE_FRONT / OUT_LIN from the environment (as front_net does).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--kind', default='boundary')
    ap.add_argument('--tag', default='arm')
    a = ap.parse_args()

    os.environ['DATA_ROOT'] = a.data
    data = FrontData(root=a.data, kinds=(a.kind,))
    net = FrontNet().to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    net.eval()

    errs = []
    with torch.no_grad():
        ki = 0
        for mi in range(data.nmedia(ki)):
            med = data.med(ki, mi)
            if len(med.TE) == 0:
                continue
            Phi = phi_of_entries(net, med, med.TE)
            ref = np.asarray(med.PHI)[med.TE]
            for c in range(len(med.TE)):
                e = 100 * np.linalg.norm((Phi[c] - ref[c])[INNER]) / max(np.linalg.norm(ref[c][INNER]), 1e-300)
                errs.append(float(e))
    e = np.asarray(errs)
    out = dict(tag=a.tag, n=int(e.size), mean=float(e.mean()), median=float(np.median(e)),
               p90=float(np.percentile(e, 90)), p99=float(np.percentile(e, 99)), max=float(e.max()),
               mode_front=MODE_FRONT, out_lin=bool(OUT_LIN), ckpt=a.ckpt)
    print(json.dumps(out, ensure_ascii=False))
    with open(os.path.join(str(RESULTS), 'eval_arm_%s.json' % a.tag), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
