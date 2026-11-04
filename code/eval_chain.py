# -*- coding: utf-8 -*-
"""端到端推理：前段背景响应 + 后段扰动响应，逐脉冲与逐场景两级 RMSPE。

用法（PowerShell，在 PAONet 根目录）：
    python code/eval_chain.py
可选环境变量：TAG、S_DIR、FRONT_W、AFTER_W、SMAX、OUTDIR。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, 'front'), os.path.join(HERE, 'after'), HERE):
    sys.path.insert(0, p)

from config import AFTER_SCENES, RESULTS, WEIGHTS          # noqa: E402
from consts import INNER, INNER_IDX, KQ, NS, WQ           # noqa: E402
import feat_torch as AF                                    # noqa: E402
from data import Medium, ray_of, scene_static, tau_ray_table  # noqa: E402
from loader import SLIM_KEYS                               # noqa: E402
from front_net import FrontNet, phi_of_entries_fast        # noqa: E402
from after_net import AfterNet, assemble                   # noqa: E402
from direct_term import FrontMed, ballistic_ids            # noqa: E402
import front_net as FN                                     # noqa: E402
import after_net as AN                                     # noqa: E402

S_DIR = os.environ.get('S_DIR', str(AFTER_SCENES))
SMAX = int(os.environ.get('SMAX', '1000'))
FRONT_W = os.environ.get('FRONT_W', str(WEIGHTS / 'formal' / 'front' / 'front_net.pt'))
AFTER_W = os.environ.get('AFTER_W', str(WEIGHTS / 'formal' / 'after' / 'after_net.pt'))
TAG = os.environ.get('TAG', 'chain')
OUTDIR = os.environ.get('OUTDIR', str(RESULTS))
os.makedirs(OUTDIR, exist_ok=True)

TEST_FRAC, SPLIT_SEED = 0.1, 20261015


def test_ids():
    ids = sorted(int(f[4:-4]) for f in os.listdir(S_DIR)
                 if f.startswith('med_') and f.endswith('.npz'))
    ids = [i for i in ids if i < SMAX]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ids))
    nte = int(round(TEST_FRAC * len(ids)))
    return sorted(ids[i] for i in perm[:nte])


def stats(a):
    a = np.asarray(a, np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                max=float(a.max()))


def main():
    t00 = time.time()
    front = FrontNet().to(FN.DEV)
    front.load_state_dict(torch.load(FRONT_W, map_location=FN.DEV))
    front.eval()
    after = AfterNet(F=int(os.environ.get('F', '192')),
                     nlayer=int(os.environ.get('NLAYER', '6')),
                     ncross=int(os.environ.get('NCROSS', '2'))).to(AN.DEV)
    after.load_state_dict(torch.load(AFTER_W, map_location=AN.DEV))
    after.eval()
    ADEV = AN.DEV

    mids = test_ids()
    print('串联推理：数据 %s ｜ 前段 %s ｜ 后段 %s ｜ 测试场景 %d'
          % (os.path.relpath(S_DIR), os.path.relpath(FRONT_W), os.path.relpath(AFTER_W), len(mids)),
          flush=True)

    per_pulse, per_scene, per_bg = [], [], []
    for k, mid in enumerate(mids):
        a = np.load(os.path.join(S_DIR, 'med_%04d.npz' % mid), allow_pickle=False)
        med = Medium.from_file(a)
        stt = scene_static(med, mid)
        IDX = np.asarray(a['IDX'], np.int64)
        PHIf = np.asarray(a['PHI'], np.float32)
        sp = np.zeros(NS * NS, np.float64)
        sr = np.zeros(NS * NS, np.float64)
        for i in range(len(IDX)):
            idx = IDX[i][None, :]
            IDb, _ = ballistic_ids(idx, med.st_bg)
            fmed = FrontMed(med.st_bg, med.w0_bg, med.g, idx, IDb)
            Phi_bg, I_bg = phi_of_entries_fast(front, fmed, np.arange(1), return_I=True)
            Itot = np.array(I_bg[0], np.float64)
            Itot[..., int(idx[0, 2])] += IDb[0]
            r0, om_raw = ray_of(IDX[i])
            omu = om_raw / max(float(np.linalg.norm(om_raw)), 1e-12)
            s_tab, tau_tab = tau_ray_table(med, r0, omu)
            Pb = np.asarray(Phi_bg[0], np.float32)
            s = dict(tokP=stt['tokP'], tokR=stt['tokR'], qryP=stt['qryP'], qryR=stt['qryR'],
                     r0=r0, om_raw=om_raw, omu=omu, s_tab=s_tab, tau_tab=tau_tab,
                     Pb=Pb, pk=np.array([max(float(Pb.max()), 1e-30)], np.float32),
                     ell=np.array([med.ell_bg], np.float32),
                     st_bg=np.array([med.st_bg], np.float32),
                     g=np.array([med.g], np.float32),
                     w0=np.array([med.w0_bg], np.float32),
                     D=np.zeros((NS * NS, KQ), np.float32),
                     Yb=np.log(np.maximum(Itot, 1e-12)).reshape(-1, KQ).astype(np.float32),
                     Phi=PHIf[i].reshape(-1))
            raw = {kk: np.expand_dims(s[kk], 0) for kk in SLIM_KEYS}
            tb = AF.build(raw, ADEV)
            with torch.no_grad():
                D = after(tb['tok'], tb['qry'], tb['bself'], tb['bcross'])
                _I, Phi = assemble(D, tb['Yb'])
            pv = Phi[0].cpu().numpy().astype(np.float64)
            rv = PHIf[i].astype(np.float64).reshape(-1)
            per_pulse.append(100.0 * np.linalg.norm((pv - rv)[INNER_IDX])
                             / max(np.linalg.norm(rv[INNER_IDX]), 1e-300))
            bgv = np.asarray(Phi_bg[0], np.float64).reshape(-1)
            per_bg.append(100.0 * np.linalg.norm((bgv - rv)[INNER_IDX])
                          / max(np.linalg.norm(rv[INNER_IDX]), 1e-300))
            sp += pv
            sr += rv
        per_scene.append(100.0 * np.linalg.norm((sp - sr).reshape(NS, NS)[INNER])
                         / max(np.linalg.norm(sr.reshape(NS, NS)[INNER]), 1e-300))
        if (k + 1) % 10 == 0 or k == 0:
            print('  %3d/%d 场景（当前场景 %04d 逐场景 ρ=%.4f%%）  %.1f min'
                  % (k + 1, len(mids), mid, per_scene[-1], (time.time() - t00) / 60), flush=True)

    rp, rs = np.asarray(per_pulse), np.asarray(per_scene)
    rb = np.asarray(per_bg)
    for nm, arr in (('逐脉冲', rp), ('逐场景', rs), ('前段背景（对照）', rb)):
        st = stats(arr)
        print('  %s %d 条 ｜ 均值 %.4f%%  中位 %.4f%%  P90 %.4f%%  P99 %.4f%%  最大 %.4f%%'
              % (nm, st['n'], st['mean'], st['median'], st['p90'], st['p99'], st['max']),
              flush=True)
    res = dict(model='PAONet（前段+后段串联）', data=os.path.relpath(S_DIR),
               front=os.path.relpath(FRONT_W), after=os.path.relpath(AFTER_W),
               pulse=stats(rp), scene=stats(rs), bgfront=stats(rb),
               minutes=round((time.time() - t00) / 60, 1))
    with open(os.path.join(OUTDIR, 'chain_%s.json' % TAG), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    np.savez(os.path.join(OUTDIR, 'chain_%s.npz' % TAG), pulse=rp, scene=rs, bg=rb,
             mid=np.asarray(mids))
    print('  用时 %.1f min -> %s' % ((time.time() - t00) / 60,
                                    os.path.join(OUTDIR, 'chain_%s.json' % TAG)), flush=True)


if __name__ == '__main__':
    main()
