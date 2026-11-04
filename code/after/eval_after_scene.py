# -*- coding: utf-8 -*-
"""Per-scene evaluation of an after-net checkpoint.

A scene's 100 pulses are summed first, then compared with the summed reference
inner field. Reads MODE_AFTER and the AB_* switches from the environment.

Usage:
    python code/after/eval_after_scene.py --ckpt weights/formal/after/after_net.pt --tag formal_after
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

from config import RESULTS                            # noqa: E402
from common import INNER_IDX                          # noqa: E402
from data import AfterData                            # noqa: E402
from after_net import DEV, MODE_AFTER, AfterNet, assemble   # noqa: E402
from feat_torch import build                          # noqa: E402
from loader import SLIM_KEYS                          # noqa: E402
from train_after import apply_mode                    # noqa: E402


def rel(a, b):
    a = np.asarray(a, np.float64).reshape(-1)[INNER_IDX]
    b = np.asarray(b, np.float64).reshape(-1)[INNER_IDX]
    return 100.0 * float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='scene')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    data = AfterData(verbose=False)
    F = int(os.environ.get('F', '192'))
    NL = int(os.environ.get('NLAYER', '6'))
    NC = int(os.environ.get('NCROSS', '2'))
    net = AfterNet(F=F, nlayer=NL, ncross=NC).to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    net.eval()

    mids = data.te[:a.limit] if a.limit else data.te
    errs = []
    with torch.no_grad():
        for mid in mids:
            n = data.n_pulse(mid)
            sp = np.zeros(41 * 41, np.float64)
            sr = np.zeros(41 * 41, np.float64)
            for i in range(n):
                s = data.sample(mid, i, slim=True)
                raw = {k: np.expand_dims(s[k], 0) for k in SLIM_KEYS}
                tb = apply_mode(build(raw, DEV))
                _I, Phi = assemble(net(tb['tok'], tb['qry'], tb['bself'], tb['bcross']), tb['Yb'])
                sp += Phi[0].cpu().numpy().astype(np.float64)
                sr += tb['Phi'][0].cpu().numpy().astype(np.float64)
            errs.append(rel(sp, sr))
            print('scene %d done' % mid, flush=True)

    e = np.asarray(errs)
    out = dict(tag=a.tag, n_scene=int(e.size), mean=float(e.mean()),
               median=float(np.median(e)), p90=float(np.percentile(e, 90)),
               p99=float(np.percentile(e, 99)), max=float(e.max()),
               mode_after=MODE_AFTER, ckpt=a.ckpt)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(str(RESULTS), exist_ok=True)
    with open(os.path.join(str(RESULTS), 'eval_after_scene_%s.json' % a.tag), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
