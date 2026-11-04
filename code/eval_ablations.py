# -*- coding: utf-8 -*-
"""Evaluate all ablation checkpoints on the per-scene protocol.

After-network cases use code/after/eval_after_scene.py.
Front-network cases use code/front/eval_front_scene.py.
Results are written to results/ablation_summary.json and results/ablation_logs/.

Usage:
    python code/eval_ablations.py
    python code/eval_ablations.py --kind after
    python code/eval_ablations.py --kind front --only L_bg,no_media
    python code/eval_ablations.py --kind after --limit 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, 'weights', 'ablations')
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(RESULTS, 'ablation_logs')

AFTER_CASES = [
    ('tab5_6/L_pt1', 'tab5_6_loss/after/L_pt1', {'MODE_AFTER': 'resid'}),
    ('tab5_6/L_pt2', 'tab5_6_loss/after/L_pt2', {'MODE_AFTER': 'logmag'}),
    ('tab5_6/L_pt3', 'tab5_6_loss/after/L_pt3', {'MODE_AFTER': 'resid'}),
    ('tab5_6/L_pt4', 'tab5_6_loss/after/L_pt4', {'MODE_AFTER': 'resid'}),
    ('tab5_6/L_pt5', 'tab5_6_loss/after/L_pt5', {'MODE_AFTER': 'resid'}),
    ('tab5_8/single_stage', 'tab5_8_structure/single_stage', {'MODE_AFTER': 'full'}),
    ('tab5_11/no_geom', 'tab5_11_feat_after/after/no_geom',
     {'MODE_AFTER': 'resid', 'AB_NOGEOM': '1'}),
    ('tab5_11/no_med', 'tab5_11_feat_after/after/no_med',
     {'MODE_AFTER': 'resid', 'AB_NOMED': '1'}),
    ('tab5_11/no_bias', 'tab5_11_feat_after/after/no_bias',
     {'MODE_AFTER': 'resid', 'AB_NOBIAS': '1'}),
]

FRONT_CASES = [
    ('tab5_6/L_bg1', 'tab5_6_loss/front/L_bg1', {}),
    ('tab5_6/L_bg2', 'tab5_6_loss/front/L_bg2', {}),
    ('tab5_6/L_bg3', 'tab5_6_loss/front/L_bg3', {'OUT_LIN': '1'}),
    ('tab5_6/L_bg4', 'tab5_6_loss/front/L_bg4', {'OUT_LIN': '1'}),
    ('tab5_7/with_prior', 'tab5_7_prior/front/with_prior', {}),
    ('tab5_7/no_prior', 'tab5_7_prior/front/no_prior', {'MODE_FRONT': 'full'}),
    ('tab5_10/formal', 'tab5_10_feat_front/front/formal', {}),
    ('tab5_10/no_media', 'tab5_10_feat_front/front/no_media', {'ZERO_COLS': '6,7,8'}),
    ('tab5_10/no_inflow', 'tab5_10_feat_front/front/no_inflow', {'ZERO_COLS': '4,5,9,10'}),
    ('tab5_10/no_query', 'tab5_10_feat_front/front/no_query',
     {'ZERO_COLS': '0,1,2,3', 'ZERO_ENC': '1'}),
    ('tab5_10/no_enc', 'tab5_10_feat_front/front/no_enc', {'ZERO_ENC': '1'}),
    ('tab5_10/no_idir', 'tab5_10_feat_front/front/no_idir', {'NO_IDIR': '1'}),
]


def _run(cmd, env_extra, log_path):
    env = dict(os.environ)
    for k in ('MODE_AFTER', 'AB_NOGEOM', 'AB_NOMED', 'AB_NOBIAS',
              'ZERO_COLS', 'ZERO_ENC', 'NO_IDIR', 'OUT_LIN', 'MODE_FRONT'):
        env.pop(k, None)
    env.update(env_extra)
    env['PYTHONIOENCODING'] = 'utf-8'
    with open(log_path, 'w', encoding='utf-8') as fh:
        return subprocess.run([sys.executable, '-u'] + cmd, cwd=ROOT, env=env,
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='after', choices=('after', 'front', 'all'))
    ap.add_argument('--only', default='', help='comma-separated name fragments')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    only = [s.strip() for s in a.only.split(',') if s.strip()]
    os.makedirs(LOGS, exist_ok=True)
    summary = {}

    if a.kind in ('after', 'all'):
        for tag, rel, envx in AFTER_CASES:
            if only and not any(s in tag for s in only):
                continue
            ckpt = os.path.join(ABL, rel, 'after_net.pt')
            if not os.path.exists(ckpt):
                print('missing', ckpt, flush=True)
                continue
            safe = tag.replace('/', '_')
            cmd = [os.path.join('code', 'after', 'eval_after_scene.py'),
                   '--ckpt', ckpt, '--tag', safe]
            if a.limit:
                cmd += ['--limit', str(a.limit)]
            envx = dict(envx, F='192', NLAYER='6', NCROSS='2')
            rc = _run(cmd, envx, os.path.join(LOGS, safe + '.log'))
            p = os.path.join(RESULTS, 'eval_after_scene_%s.json' % safe)
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    summary[tag] = json.load(fh)
            print('%-22s rc=%d' % (tag, rc), flush=True)

    if a.kind in ('front', 'all'):
        data = os.path.join(ROOT, 'data', 'front', 'data1')
        for tag, rel, envx in FRONT_CASES:
            if only and not any(s in tag for s in only):
                continue
            ckpt = os.path.join(ABL, rel, 'front_net.pt')
            if not os.path.exists(ckpt):
                print('missing', ckpt, flush=True)
                continue
            safe = tag.replace('/', '_')
            cmd = [os.path.join('code', 'front', 'eval_front_scene.py'),
                   '--ckpt', ckpt, '--data', data,
                   '--kind', 'boundary', '--tag', safe]
            rc = _run(cmd, envx, os.path.join(LOGS, safe + '.log'))
            p = os.path.join(RESULTS, 'eval_front_scene_%s.json' % safe)
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    summary[tag] = json.load(fh)
            print('%-22s rc=%d' % (tag, rc), flush=True)

    out = os.path.join(RESULTS, 'ablation_summary.json')
    if os.path.exists(out):
        try:
            with open(out, encoding='utf-8') as fh:
                old = json.load(fh)
            old.update(summary)
            summary = old
        except Exception:
            pass
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print('->', out, flush=True)


if __name__ == '__main__':
    main()
