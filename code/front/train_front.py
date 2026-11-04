# -*- coding: utf-8 -*-
"""PAONet 前段（背景响应网络）训练与评估。

数据：data/front/data1 下的边界单位脉冲样本；按介质做场景级 9:1 划分，
      测试场景的全部脉冲只用于测试。
损失：加权 log 残差 MSE，权重 sqrt(Itgt/(Itgt+1e-7))（弱场处降权）。
优化：Adam（不做权重衰减、不启用梯度裁剪）+ 余弦退火到 eta_min；初始化与采样种子固定。

效率要点
  · 每步把整批（可跨介质、跨类）组装成一个张量，**只做一次前向/反向**；
    此前按介质分组各做一次前向，单次只有千余行，GPU 利用率极低。
  · 散射算子按 g 预存为设备张量，按行索引取用（每步复用预存相函数矩阵）。
  · 验证批与验证条目固定，张量只构造一次，后续验证直接复用。
  · 单精度矩阵乘启用 TF32。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from common import INNER, KQ, NS, WQ, new_run_dir
from data import FrontData
from front_net import DEV, D_LAT, NBLK, NSCAT, WIDE, FrontNet, build_batch, phi_of_entries, pk_stack

NSTEP = int(os.environ.get('NSTEP', '100000'))   # 可由环境变量覆盖
TPER = 60000                   # 退火周期（硬编码；与总步数解耦，超出后保持周期末学习率）
NGRP = 273                     # 每步抽取的 (介质, 条目) 对数
NPT = 1                        # 每个对抽取的查询节点数（每步 = 273 对 × 1 节点 × 24 方向）
LR = float(os.environ.get('LR_FRONT', '2e-3'))
WD = float(os.environ.get('WD_FRONT', '0'))     # 正式版：Adam，无权重衰减（0 表示关闭）
ETA_MIN = float(os.environ.get('ETA_FRONT', '1e-5'))
SEED = int(os.environ.get('SEED_FRONT', '1'))   # 可用 SEED_FRONT 改；默认 1，与既有训练一致
VAL_EVERY = 2000               # 验证间隔
VAL_BATCHES = 4                # 验证用的固定批数
VAL_ENTRIES = 2                # 每个介质用于验证 Φ 的测试条目数


def lr_at(step):
    """余弦退火：周期 TPER 步降到位；**超出退火周期后固定为周期末的学习率**。"""
    t = min(step, TPER)
    return ETA_MIN + (LR - ETA_MIN) * (1.0 + np.cos(np.pi * t / TPER)) / 2.0


def build_batch_full(data, pkt, sel, ii, jj):
    """整批组装成一个批次（跨介质、跨类），返回设备上的 (blk, tgt, wgt, rel, pk)。

    pkt 为 (G,KQ,KQ) 的散射算子栈；每行按其所属介质的 g 下标索引，得到 (N,KQ,KQ)。
    """
    N = len(sel) * NPT
    blk = np.empty((N, KQ, NBLK), np.float32)
    tgt = np.empty((N, KQ), np.float32)
    wgt = np.empty((N, KQ), np.float32)
    rel = np.empty((N, KQ, 2), np.float32)
    gi = np.empty(N, np.int64)
    ai_row = np.repeat(sel[:, 2], NPT)
    grow = np.repeat(sel[:, 0] * 1000 + sel[:, 1], NPT)
    kk = np.tile(np.arange(KQ), (NPT, 1))
    for g in np.unique(grow):
        ki, mi = int(g) // 1000, int(g) % 1000
        med = data.med(ki, mi)
        m = grow == g
        n = int(m.sum())
        b, t, w, r = build_batch(med, ai_row[m], ii[m], jj[m],
                                 np.tile(np.arange(KQ), (n, 1)))
        blk[m] = b; tgt[m] = t; wgt[m] = w; rel[m] = r
        gi[m] = data.gidx[data.kinds[ki]][mi]
    return (torch.tensor(blk, device=DEV), torch.tensor(tgt, device=DEV),
            torch.tensor(wgt, device=DEV), torch.tensor(rel, device=DEV),
            torch.tensor(np.asarray(data.gset, np.float32)[gi], device=DEV))


LOSS_FRONT = os.environ.get('LOSS_FRONT', 'log').lower()   # log|logp|lin|twopart
W_ANG = float(os.environ.get('W_ANG', '0.1'))
INIT_LOG = float(os.environ.get('INIT_LOG', '-900'))
CLIP = float(os.environ.get('CLIP', '0'))       # 正式版：不启用梯度裁剪（<=0 即不裁剪）
EPS = 1e-12
COMP = {}


def batch_mse(net, batch):
    blk, tgt, wgt, rel, gvals = batch
    o = net(blk, rel, gvals)
    if LOSS_FRONT in ('logp', 'log_plain'):
        return ((o - tgt) ** 2).sum() / tgt.numel()
    if LOSS_FRONT == 'lin':
        return ((torch.exp(o) - torch.exp(tgt)) ** 2).sum() / tgt.numel()
    if LOSS_FRONT == 'twopart':
        wq = torch.as_tensor(WQ, dtype=torch.float32, device=o.device)
        ID = torch.exp(blk[..., 11].clamp(min=-60.0, max=60.0))
        I_ref = ID + torch.exp(tgt.clamp(max=60.0))
        I_hat = ID + torch.exp(o.clamp(max=60.0))
        Phi_hat = I_hat @ wq
        Phi_ref = I_ref @ wq
        den = Phi_ref.abs()
        l_phi = ((Phi_hat - Phi_ref).abs() / den.clamp_min(1e-30)).mean()
        d = (torch.log(I_hat.clamp_min(EPS)) - torch.log(I_ref.clamp_min(EPS))).abs()
        g = (I_ref / I_ref.amax(dim=1, keepdim=True).clamp_min(1e-30)).clamp(0.0, 1.0)
        ww = g * wq[None, :]
        l_ang = ((d * ww).sum(1) / ww.sum(1).clamp_min(1e-30)).mean()
        COMP['phi'] = float(l_phi)
        COMP['ang'] = float(l_ang)
        return l_phi + W_ANG * l_ang
    return (wgt * (o - tgt) ** 2).sum() / tgt.numel()


def sample_pairs(data, rng, n):
    med = [(ki, mi) for ki in range(len(data.kinds))
           for mi in range(data.nmedia(ki))
           if len(data.media[data.kinds[ki]][mi].TR) > 0]
    nblk = min(10, n, len(med))          # 小数据集（如附带子集）下也能跑
    pick = rng.choice(len(med), size=nblk, replace=False)
    rows = []
    for j, q in enumerate(pick):
        ki, mi = med[int(q)]
        tr = data.media[data.kinds[ki]][mi].TR
        cnt = n // nblk + (1 if j < n % nblk else 0)
        rows.append(np.stack([np.full(cnt, ki), np.full(cnt, mi), rng.choice(tr, cnt)], -1))
    return np.concatenate(rows, 0).astype(np.int64)


def make_val(data, pkt):
    """固定验证批（预建张量，复用）与固定验证条目。"""
    rng = np.random.default_rng(7)
    batches = []
    for _ in range(VAL_BATCHES):
        sel = sample_pairs(data, rng, NGRP)
        batches.append(build_batch_full(data, pkt, sel,
                                        rng.integers(0, NS, NGRP * NPT),
                                        rng.integers(0, NS, NGRP * NPT)))
    ents = []
    for ki in range(len(data.kinds)):
        for mi in range(min(50, data.nmedia(ki))):   # 验证只取固定 50 个介质
            med = data.med(ki, mi)
            if len(med.TE):
                s = np.random.default_rng(11).choice(
                    med.TE, size=min(VAL_ENTRIES, len(med.TE)), replace=False)
                ents.append((ki, mi, np.sort(s)))
    return batches, ents


def val_phi(net, data, ents):
    """固定验证条目的 Φ 相对误差，按类返回中位数与均值（%）。"""
    out = {}
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for ki, k in enumerate(data.kinds):
            errs = []
            for (kk, mi, idx) in ents:
                if kk != ki:
                    continue
                med = data.med(ki, mi)
                Phi = phi_of_entries(net, med, idx)
                ref = np.asarray(med.PHI)[idx]
                for c in range(len(idx)):
                    d = ref[c][INNER]
                    errs.append(100 * np.linalg.norm((Phi[c] - ref[c])[INNER])
                                / max(np.linalg.norm(d), 1e-300))
            if errs:
                out[k] = (float(np.median(errs)), float(np.mean(errs)))
    if was_training:
        net.train()
    return out


def train(data):
    torch.manual_seed(SEED)
    net = FrontNet().to(DEV)
    npar = sum(p.numel() for p in net.parameters())
    pkt = pk_stack(data.gset)
    run_dir = os.environ.get('OUTDIR') or new_run_dir()
    os.makedirs(run_dir, exist_ok=True)                       # 本次训练专属目录 train1 / train2 / ...
    ckpt = os.path.join(run_dir, 'front_net.pt')
    print('本次训练权重目录：%s' % run_dir)
    print('数据：%s' % data.root)
    for k in data.kinds:
        ne = data.media[k][0].NENT
        print('  %-9s %d 个介质 × %d 条基元源（测试条目取自各文件 TE）'
              % (k, len(data.media[k]), ne))
    print('训练候选 (类,介质,条目) %d 个；主干 %d^3 -> %d -> [%d 散射块]；参数 %d'
          % (len(data.rows), WIDE, D_LAT, NSCAT, npar))
    print('每步 %d 对 × %d 节点 × %d 方向 = %d 个目标（一次前向）；步数 %d；种子 %d；TF32 %s'
          % (NGRP, NPT, KQ, NGRP * NPT * KQ, NSTEP, SEED,
             bool(torch.backends.cuda.matmul.allow_tf32)), flush=True)
    if WD > 0:
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        print('  优化器 AdamW（weight_decay=%.3g）' % WD, flush=True)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=LR)      # 完全关闭权重衰减
        print('  优化器 Adam（不做权重衰减）', flush=True)
    if INIT_LOG > -900:                      # 把网络输出（对数残差）放到典型量级上
        with torch.no_grad():
            last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
            if last.bias is not None:
                last.bias.fill_(INIT_LOG)
                last.weight.mul_(0.01)
        print('  输出头偏置初始化为 %.3f（e^o = %.2e）' % (INIT_LOG, float(torch.exp(torch.tensor(INIT_LOG)))))
    # 学习率：余弦退火，周期 TPER 步；**超出退火周期后固定为周期末的学习率**（见 lr_at）
    print('  退火周期 %d 步（硬编码，与总步数解耦）：%.2e -> %.2e；%d 步之后固定 %.2e'
          % (TPER, LR, ETA_MIN, TPER, lr_at(TPER)), flush=True)
    batches, ents = make_val(data, pkt)
    rng = np.random.default_rng(1)
    ema = None
    t0 = time.time(); tprev = t0
    hist = dict(tr_step=[], tr_loss=[], tr_ema=[], va_step=[], va_med=[], va_mean=[])
    for step in range(1, NSTEP + 1):
        sel = sample_pairs(data, rng, NGRP)
        ii = rng.integers(0, NS, NGRP * NPT)
        jj = rng.integers(0, NS, NGRP * NPT)
        batch = build_batch_full(data, pkt, sel, ii, jj)
        opt.zero_grad(set_to_none=True)
        loss = batch_mse(net, batch)
        loss.backward()
        if CLIP > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        for g in opt.param_groups:                 # 下一步的学习率（超出周期后保持不变）
            g['lr'] = lr_at(step)
        lv = float(loss.item())
        hist['tr_step'].append(step); hist['tr_loss'].append(lv)
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
        hist['tr_ema'].append(float(ema))
        if step % max(1, NSTEP // 30) == 0 or step == 1:
            now = time.time()
            extra = ''
            if LOSS_FRONT == 'twopart' and COMP:
                extra = '  [角积分 %.4e ｜ 逐方向 %.4e]' % (COMP['phi'], COMP['ang'])
            print('    step %6d  loss %.6f (EMA %.6f)  lr %.2e  (%.1f s, 累计 %.1f min)%s'
                  % (step, lv, ema, opt.param_groups[0]['lr'], now - tprev, (now - t0) / 60, extra),
                  flush=True)
            tprev = now
        if VAL_EVERY and (step % VAL_EVERY == 0 or step == NSTEP):
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([float(batch_mse(net, b)) for b in batches]))
            vp = val_phi(net, data, ents)
            net.train()
            for k, v in vp.items():
                hist['va_step'].append(step); hist['va_med'].append(v[0]); hist['va_mean'].append(v[1])
            print('      [验证] step %6d  测试MSE %.3e（训练/测试 %.2f）  测试Φ 中位 %s  均值 %s'
                  % (step, vl, lv / max(vl, 1e-30),
                     '  '.join('%s %.4f%%' % (k, v[0]) for k, v in vp.items()),
                     '  '.join('%s %.4f%%' % (k, v[1]) for k, v in vp.items())), flush=True)
        if step % 10000 == 0:
            torch.save(net.state_dict(), ckpt)
    torch.save(net.state_dict(), ckpt)
    np.savez(os.path.join(os.path.dirname(ckpt), 'history.npz'),
             tr_step=np.asarray(hist['tr_step'], np.int32),
             tr_loss=np.asarray(hist['tr_loss'], np.float64),
             tr_ema=np.asarray(hist['tr_ema'], np.float64),
             va_step=np.asarray(hist['va_step'], np.int32),
             va_med=np.asarray(hist['va_med'], np.float64),
             va_mean=np.asarray(hist['va_mean'], np.float64))
    print('训练完成，权重 %s（%.1f min）' % (ckpt, (time.time() - t0) / 60), flush=True)
    return net


def evaluate(net, data):
    """按类逐介质报逐条目 Φ 相对 L2（内区）。"""
    net.eval()
    t0 = time.time()
    print('\n评估：逐条目 Φ 相对 L2（内区 39×39）', flush=True)
    with torch.no_grad():
        for ki, k in enumerate(data.kinds):
            all_te = []
            print('  [%s]' % k)
            for mi in [j for j in range(data.nmedia(ki)) if len(data.med(ki, j).TE) > 0]:   # 只用测试场景   # 验证只取固定 50 个介质
                med = data.med(ki, mi)
                Phi = phi_of_entries(net, med, med.TE)
                ref = np.asarray(med.PHI)[med.TE]
                r = np.array([100 * np.linalg.norm((Phi[c] - ref[c])[INNER])
                              / max(np.linalg.norm(ref[c][INNER]), 1e-300)
                              for c in range(len(med.TE))])
                all_te.append(r)
                print('    med_%02d (σ_t=%.0f ω₀=%.2f g=%.2f)  测试 %4d 条  中位 %7.4f%%  '
                      'P90 %7.4f%%  最大 %8.4f%%'
                      % (med.mid, med.st, med.w0, med.g, len(r), np.median(r),
                         np.percentile(r, 90), r.max()), flush=True)
            r = np.concatenate(all_te)
            print('    ── %s 汇总：中位 %7.4f%%  P90 %7.4f%%  最大 %8.4f%%'
                  % (k, np.median(r), np.percentile(r, 90), r.max()), flush=True)
    print('评估用时 %.1f min' % ((time.time() - t0) / 60), flush=True)


def main():
    data = FrontData()
    net = train(data)
    evaluate(net, data)


if __name__ == '__main__':
    main()
