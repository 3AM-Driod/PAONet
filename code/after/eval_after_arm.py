# -*- coding: utf-8 -*-
"""Full-test-set evaluation for an after-net checkpoint (100 test scenes x 100 pulses = 1e4).

Reads MODE_AFTER from the environment, exactly as training does.
Reports mean / median / P90 / P99 / max of the per-pulse relative L2 error of Phi.
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
from common import EPS, INNER_IDX                     # noqa: E402
from data import AfterData                            # noqa: E402
from after_net import DEV, MODE_AFTER, AfterNet, assemble   # noqa: E402
from feat_torch import build                          # noqa: E402
from loader import SLIM_KEYS                          # noqa: E402
from train_after import apply_mode                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='arm')
    a = ap.parse_args()

    data = AfterData(verbose=False)
    F = int(os.environ.get('F', '192'))
    NL = int(os.environ.get('NLAYER', '6'))
    NC = int(os.environ.get('NCROSS', '2'))
    net = AfterNet(F=F, nlayer=NL, ncross=NC).to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    net.eval()

    errs = []
    with torch.no_grad():
        for mid in data.te:
            n = data.n_pulse(mid)
            for i in range(n):
                s = data.sample(mid, i, slim=True)
                raw = {k: np.expand_dims(s[k], 0) for k in SLIM_KEYS}
                tb = apply_mode(build(raw, DEV))
                _I, Phi = assemble(net(tb['tok'], tb['qry'], tb['bself'], tb['bcross']), tb['Yb'])
                p = Phi[0].cpu().numpy()
                t = tb['Phi'][0].cpu().numpy()
                errs.append(float(np.linalg.norm((p - t)[INNER_IDX])
                                  / max(np.linalg.norm(t[INNER_IDX]), 1e-30)))
            print('scene %d done, %d pulses' % (mid, len(errs)), flush=True)

    e = np.asarray(errs) * 100.0
    out = dict(tag=a.tag, n=int(e.size), mean=float(e.mean()), median=float(np.median(e)),
               p90=float(np.percentile(e, 90)), p99=float(np.percentile(e, 99)),
               max=float(e.max()), mode_after=MODE_AFTER, ckpt=a.ckpt)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open(os.path.join(str(RESULTS), 'eval_after_%s.json' % a.tag), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
