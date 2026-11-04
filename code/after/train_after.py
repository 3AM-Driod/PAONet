# -*- coding: utf-8 -*-
"""PAONet 后段（扰动响应网络）训练脚本（块采样 + 验证 + 权重目录）。

用法（PowerShell，在 PAONet 根目录）：
    python code\\after\\train_after.py

环境变量：F、NLAYER、NCROSS、NSTEP、TPER、LR、ETA_MIN、BATCH、NSC（每步场景数）、
          LOG、VAL、W_ANG、CKPT（续训）、DRY（冒烟：只跑若干步）
          消融开关（默认全关）：AB_NOGEOM（去射线几何）、AB_NOMED（去介质读数）、
          AB_NOBIAS（去唯一偏置 τ_ij）
          辅助项加权：AW=bg（默认，背景角向光强）| full（完整角向光强）
          余弦退火周期由 TPER 决定、与 NSTEP 无关：缩短总步数不会改变任一步的学习率

损失
    主：内区（39×39）逐样本 Φ 相对 L2；Φ = Σ_k w_k I_k，I = (I_背景+EPS)·exp(D_θ) − EPS
    辅：对数比残差 D 的内区角度加权 L1（跨数量级更稳）
学习目标 = 纯背景响应残差（不含任何解析弹道项）
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import AB, EPS, INNER_IDX, NBIAS, NFEAT, WQ, ab_name, new_run_dir   # noqa: E402
from data import AfterData                                              # noqa: E402
from feat_torch import build                                            # noqa: E402
from loader import SLIM_KEYS, make_loader                               # noqa: E402
from after_net import DEV, AfterNet, assemble                           # noqa: E402

F = int(os.environ.get('F', '192'))
NLAYER = int(os.environ.get('NLAYER', '6'))
NCROSS = int(os.environ.get('NCROSS', '2'))
NSTEP = int(os.environ.get('NSTEP', '100000'))
TPER = int(os.environ.get('TPER', '60000'))
LR = float(os.environ.get('LR', '3e-4'))
ETA_MIN = float(os.environ.get('ETA_MIN', '3e-5'))
BATCH = int(os.environ.get('BATCH', '16'))
NSC = int(os.environ.get('NSC', '2'))
NWORKERS = int(os.environ.get('NWORKERS', '6'))
LOG = int(os.environ.get('LOG', '100'))
VAL = int(os.environ.get('VAL', '2000'))
W_ANG = float(os.environ.get('W_ANG', '0.1'))
AW = os.environ.get('AW', 'bg')                 # 辅助项的角向权重取哪一支：bg（背景场）| full（完整场）
DRY = int(os.environ.get('DRY', '0'))
GPUFEAT = os.environ.get('GPUFEAT', '1') == '1'
MODE = os.environ.get('MODE', 'resid')          # resid：学残差（现行）；full：直接学整场
WD = float(os.environ.get('WD_AFTER', '0'))         # 正式版：Adam，无权重衰减（0 表示关闭）
CLIP = float(os.environ.get('CLIP_AFTER', '0'))     # 正式版：不启用梯度裁剪（<=0 表示不裁剪）

WQ_T = torch.tensor(WQ, dtype=torch.float32, device=DEV)
II = torch.tensor(INNER_IDX, dtype=torch.long, device=DEV)
KEYS = ('tok', 'qry', 'bself', 'bcross', 'D', 'Yb', 'Phi')


def lr_at(step):
    if step >= TPER:
        return ETA_MIN
    return ETA_MIN + (LR - ETA_MIN) * 0.5 * (1.0 + np.cos(np.pi * step / TPER))


def batch_of(data, mids, npulse_each, rng):
    """按场景块构造一个 batch。

    GPUFEAT=1（默认）：CPU 只做记忆映射切片（小量数据），
    逐点特征与两套物理偏置由 feat_torch 在 GPU 上批量构建。
    GPUFEAT=0：使用 CPU（numpy）路径。
    """
    items = []
    for mid in mids:
        nt = len(np.asarray(data._scene(mid)['a']['IDX']))
        for i in rng.choice(nt, size=min(npulse_each, nt), replace=False):
            items.append(data.sample(mid, int(i), slim=GPUFEAT))
    if not GPUFEAT:
        return {k: torch.cat([torch.as_tensor(it[k], dtype=torch.float32, device=DEV)
                              .unsqueeze(0) for it in items], 0) for k in KEYS}
    import feat_torch
    keys = ('tokP', 'tokR', 'qryP', 'qryR', 'r0', 'om_raw', 'omu', 's_tab', 'tau_tab',
            'ell', 'st_bg', 'g', 'D', 'Yb', 'Phi')
    s = {k: np.stack([it[k] for it in items], 0) for k in keys}
    return feat_torch.build(s, DEV)


def apply_mode(tb):
    """MODE=full：目标换成整场对数（D+Yb），装配基准置零（I=exp(D)）——即"直接学整场"。"""
    if MODE == 'full':
        tb['D'] = tb['D'] + tb['Yb']
        tb['Yb'] = torch.zeros_like(tb['Yb'])
    return tb


LOSS_AFTER = os.environ.get('LOSS_AFTER', 'formal').lower()
REL_EPS = float(os.environ.get('REL_EPS', '1e-12'))
PHI_LAMBDA = float(os.environ.get('PHI_LAMBDA', '1'))


def loss_of(net, tb):
    D = net(tb['tok'], tb['qry'], tb['bself'], tb['bcross'])
    if LOSS_AFTER in ('b1_logmag', 'b2_lin'):
        Yb = tb['Yb']
        Dref = tb['D']
        Ib = (torch.exp(Yb.clamp(max=60.0)) - EPS).clamp_min(0.0)
        Pt = (Ib + EPS) * torch.expm1(Dref.clamp(min=-60.0, max=60.0))
        w = Pt.abs() / (Pt.abs().amax(dim=-1, keepdim=True) + REL_EPS)
        if LOSS_AFTER == 'b1_logmag':
            tgt = torch.log(Pt.abs() + EPS)
            res = (D - tgt)[:, II, :]
        else:
            res = (D - Pt)[:, II, :]
        l_log = (w[:, II, :] * res ** 2).sum() / res.numel()
        _I, Phi_h = assemble(D, tb['Yb'])
        Phi_ref = tb['Phi']
        num = torch.linalg.norm((Phi_h - Phi_ref)[:, II], dim=1)
        den = torch.linalg.norm(Phi_ref[:, II], dim=1).clamp_min(1e-30)
        l_phi = (num / den).mean()
        loss = l_log + PHI_LAMBDA * l_phi
        return loss, float(l_phi), float(l_log)
    if LOSS_AFTER == 'relpphi3':
        Yb = tb['Yb']
        Dref = tb['D']
        Ib = (torch.exp(Yb.clamp(max=60.0)) - EPS).clamp_min(0.0)
        Pt = (Ib + EPS) * torch.expm1(Dref.clamp(min=-60.0, max=60.0))
        Ph = (Ib + EPS) * torch.expm1(D.clamp(min=-60.0, max=60.0))
        w = Pt.abs()
        w = w / (w.amax(dim=-1, keepdim=True) + REL_EPS)
        d = (D - Dref)[:, II, :]
        l_log = (w[:, II, :] * d ** 2).sum() / d.numel()
        Phi_p = (Pt * WQ_T).sum(-1)
        Phi_h = (Ph * WQ_T).sum(-1)
        Phi_ref = ((Ib + Pt) * WQ_T).sum(-1)
        num = torch.linalg.norm((Phi_h - Phi_p)[:, II], dim=1)
        den = torch.linalg.norm(Phi_ref[:, II], dim=1).clamp_min(1e-30)
        l_phi = (num / den).mean()
        loss = l_log + PHI_LAMBDA * l_phi
        return loss, float(l_phi), float(l_log)
    if LOSS_AFTER == 'msed':
        m = ((D - tb['D'])[:, II, :] ** 2).mean()
        return m, float(m), 0.0
    _I, Phi = assemble(D, tb['Yb'])
    num = (Phi - tb['Phi'])[:, II]
    den = tb['Phi'][:, II]
    l_phi = (torch.linalg.norm(num, dim=1)
             / torch.linalg.norm(den, dim=1).clamp_min(1e-30)).mean()
    d = (D - tb['D'])[:, II, :]
    if AW == 'full':
        Iref = torch.exp((tb['Yb'] + tb['D'])[:, II, :].clamp(max=60.0)) - EPS
    else:
        Iref = torch.exp(tb['Yb'][:, II, :].clamp(max=60.0)) - EPS
    wgt = (Iref / Iref.amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)).clamp(0.0, 1.0)
    ww = (WQ_T[None, None, :] * wgt)
    l_ang = ((d.abs() * ww).sum(-1) / ww.sum(-1).clamp_min(1e-30)).mean()
    return l_phi + W_ANG * l_ang, float(l_phi), float(l_ang)


@torch.no_grad()
def evaluate(net, data, pairs):
    """返回 (均值, 中位, P90, 最大)：均值与训练损失口径一致，可直接与训练曲线比较。"""
    net.eval()
    out = []
    for mid, i in pairs:
        s = data.sample(mid, i, slim=True)
        raw = {k: np.expand_dims(s[k], 0) for k in SLIM_KEYS}
        tb = apply_mode(build(raw, DEV))
        _I, Phi = assemble(net(tb['tok'], tb['qry'], tb['bself'], tb['bcross']), tb['Yb'])
        p = Phi[0].cpu().numpy()
        t = tb['Phi'][0].cpu().numpy()
        out.append(np.linalg.norm((p - t)[INNER_IDX])
                   / max(np.linalg.norm(t[INNER_IDX]), 1e-30))
    net.train()
    o = np.array(out)
    return (float(o.mean()), float(np.median(o)), float(np.percentile(o, 90)), float(o.max()))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    data = AfterData()
    net = AfterNet(F=F, nlayer=NLAYER, ncross=NCROSS).to(DEV)
    npar = sum(p.numel() for p in net.parameters())
    per = max(1, BATCH // NSC)
    run = new_run_dir()
    print('run 目录 %s ｜ 参数量 %.2f M ｜ F=%d 层=%d 交叉=%d ｜ 每步 %d 场景 x %d 脉冲'
          % (run, npar / 1e6, F, NLAYER, NCROSS, NSC, per), flush=True)
    # 逐轮次记录完整配置，便于消融表与日志互换
    cfg = dict(F=F, NLAYER=NLAYER, NCROSS=NCROSS, NFEAT=NFEAT, NBIAS=NBIAS,
               NSTEP=NSTEP, TPER=TPER, LR=LR,
               ETA_MIN=ETA_MIN, BATCH=BATCH, NSC=NSC, NWORKERS=NWORKERS, W_ANG=W_ANG,
               AW=AW, MODE=MODE, AB=AB,
               ab_name=ab_name(), VAL=VAL, VAL_PAIRS=int(os.environ.get('VAL_PAIRS', '100')),
               SMAX=os.environ.get('SMAX', '0'), S_DIR=os.environ.get('S_DIR', ''),
               CACHE=os.environ.get('CACHE', ''), lr_end=lr_at(min(NSTEP, TPER)))
    with open(os.path.join(run, 'cfg.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print('偏置配置 %s ｜ 特征 %d 维、偏置 %d 项 ｜ 辅助项权重取 %s（W_ANG=%.3g）｜ 步数 %d ｜ '
          '余弦周期 TPER %d（与总步数无关）｜ 末步 lr %.2e'
          % (ab_name(), NFEAT, NBIAS, '完整场' if AW == 'full' else '背景场', W_ANG,
             NSTEP, TPER, cfg['lr_end']), flush=True)
    if os.environ.get('CKPT'):
        net.load_state_dict(torch.load(os.environ['CKPT'], map_location=DEV))
        print('已载入 %s' % os.environ['CKPT'], flush=True)
    if WD > 0:
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        print('优化器 AdamW（weight_decay=%.3g）' % WD, flush=True)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=LR)          # 完全关闭权重衰减
        print('优化器 Adam（不做权重衰减）', flush=True)
    rng = np.random.default_rng(12345)
    nvp = int(os.environ.get('VAL_PAIRS', '100'))
    val_pairs = [(data.te[i % len(data.te)], int(rng.integers(0, 100))) for i in range(nvp)]
    mn, med, p90, mx = evaluate(net, data, val_pairs)
    print('起始（零残差基线）：均值 %.4f%%  中位 %.4f%%  P90 %.4f%%  最大 %.4f%%（%d 个验证脉冲）'
          % (100 * mn, 100 * med, 100 * p90, 100 * mx, nvp), flush=True)
    # 曲线数据（训练逐步、验证按 VAL 步）；验证以**均值**为主（与训练损失同口径）
    hist = {k: [] for k in ('step', 'loss', 'l_phi', 'l_ang', 'lr', 'sec')}
    vhist = {k: [] for k in ('step', 'mean', 'med', 'p90', 'max')}
    vhist['step'].append(0); vhist['mean'].append(mn); vhist['med'].append(med)
    vhist['p90'].append(p90); vhist['max'].append(mx)
    hist_path = os.path.join(run, 'history.npz')

    def save_hist():
        np.savez(hist_path, **{('tr_' + k): np.asarray(v) for k, v in hist.items()},
                 **{('va_' + k): np.asarray(v) for k, v in vhist.items()},
                 meta=np.array([F, NLAYER, NCROSS, BATCH, NWORKERS, nvp]))

    t0 = time.time()
    t_start = time.time()
    loader = make_loader(data.tr, BATCH, nworkers=NWORKERS, seed=7)
    it = iter(loader)
    nstep = NSTEP if not DRY else min(NSTEP, DRY)
    for step in range(1, nstep + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        try:
            raw = next(it)
        except StopIteration:
            it = iter(loader)
            raw = next(it)
        tb = build(raw, DEV)
        loss, l_phi, l_ang = loss_of(net, apply_mode(tb))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if CLIP > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        hist['step'].append(step); hist['loss'].append(float(loss))
        hist['l_phi'].append(l_phi); hist['l_ang'].append(l_ang)
        hist['lr'].append(lr_at(step)); hist['sec'].append(time.time() - t_start)
        if step % LOG == 0 or step == 1:
            dt = (time.time() - t0) / (LOG if step > 1 else 1)
            t0 = time.time()
            print('步 %6d  损失 %.4e（Phi %.4e，角向 %.4e）  lr %.2e  %.3f s/步  batch %d'
                  % (step, float(loss), l_phi, l_ang, lr_at(step), dt, BATCH), flush=True)
        if step % VAL == 0 or step == nstep:
            mn, med, p90, mx = evaluate(net, data, val_pairs)
            vhist['step'].append(step); vhist['mean'].append(mn); vhist['med'].append(med)
            vhist['p90'].append(p90); vhist['max'].append(mx)
            print('        验证（%d 个测试脉冲）：均值 %.4f%%  中位 %.4f%%  P90 %.4f%%  最大 %.4f%%'
                  % (len(val_pairs), 100 * mn, 100 * med, 100 * p90, 100 * mx), flush=True)
            torch.save(net.state_dict(), os.path.join(run, 'after_net.pt'))
            save_hist()
    torch.save(net.state_dict(), os.path.join(run, 'after_net.pt'))
    save_hist()
    print('训练结束，权重 %s ｜ 曲线数据 %s' % (os.path.join(run, 'after_net.pt'), hist_path),
          flush=True)


if __name__ == '__main__':
    main()
