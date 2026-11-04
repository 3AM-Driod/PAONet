# -*- coding: utf-8 -*-
"""前段网络：Green 函数核的最小实现。

结构由以下面向强各向异性（g→1）的物理模块组成：
  ① **Legendre 谱角向层**：散射块里的角向算子采用 Legendre 谱展开，而非直接使用给定的 HG 核，
     A(g) = rowsum1( Σ_l (g^l · m_l) (2l+1) P_l(Ω·Ω') · ω )，m_l 为可学习逐模态倍数（初值 1）。
     由于 HG 核的 Legendre 矩恰为 g^l，**初值即精确 HG 算子**，网络只学逐模态修正；
     前向峰的角向结构由结构给定，不必从数据里学。
  ② **一次散射源先验**：特征里增加 log(μ_s · ω_{k_src} · K[k, k_src] · ID + 1e-30)，
     即一次碰撞源在查询方向上的解析强度（直接项已知、碰撞算子给定），
     让网络不必从零推测前向峰内的一次散射贡献。

其余：输入 = Fourier(rel，7 尺度 × 16 × sin/cos = 224) + 13 维物理特征 = 237 维；
主干 → 512³ → 64（GELU）；2 个散射块；输出为**散射残差的对数**，直接项既作特征又加回。
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from common import H, KQ, NS, VC, WQ, hg_kernel, pk_of

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MODE_FRONT = os.environ.get('MODE_FRONT', 'resid')   # resid｜full
OUT_LIN = os.environ.get('OUT_LIN', '0') == '1'   # True：网络输出即线性 I_sca（不做 exp）
NO_IDIR = os.environ.get('NO_IDIR', '0') == '1'   # True：特征中排除弹道项 log I_dir


def _zero_cols():
    v = os.environ.get('ZERO_COLS', '').strip()   # 正式版 236 维输入不置零；消融时用 ZERO_COLS 指定列
    return tuple(int(t) for t in v.split(',') if t.strip() != '')


ZERO_COLS = _zero_cols()   # 需要置零的特征列（F1-F6 消融用）
ZERO_ENC = os.environ.get('ZERO_ENC', '0') == '1'   # True：把 224 维傅里叶编码置零


def _apply_zero_cols(blk):
    """把指定列置零；列号 >= 12 时作用于傅里叶编码块。"""
    for c in ZERO_COLS:
        if c < blk.shape[-1]:
            blk[..., c] = 0.0
    return blk
NF = 16
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
WIDE = int(os.environ.get('FW', '384'))      # 主干宽度（容量扫描用）
D_LAT = int(os.environ.get('FL', '64'))      # 隐层/角向块宽度
NSCAT = int(os.environ.get('NSC_FRONT', '2'))   # 散射块层数；默认 2，用 NSC_FRONT 可改为 3 等
LB_ORDER = 6
NBLK = 12                              # 正式版：移除一次散射源先验列
SIG_MAX = 16.0                         # 与后段一致的消光归一常数（表 5.1）
LOG_EPS = -27.631021115928547          # log(1e-12)
LOG_EPS2 = -69.07755278982137          # log(1e-30)

if DEV == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_KCACHE = {}


def pk_stack(gs):
    """（保留：外部若需按 g 取给定 HG 算子可用）"""
    return torch.stack([torch.tensor(pk_of(float(g)), device=DEV) for g in gs])


def legendre_basis(L=LB_ORDER):
    """B_l[i,j] = (2l+1) P_l(Ω_i·Ω_j)。"""
    vn = np.linalg.norm(VC, axis=1)
    O = np.concatenate([VC, np.sqrt(np.maximum(1.0 - vn ** 2, 0.0))[:, None]], -1)
    mu = np.clip(O @ O.T, -1.0, 1.0)
    B = np.empty((L + 1, KQ, KQ), np.float32)
    for l in range(L + 1):
        B[l] = (2 * l + 1) * np.polynomial.legendre.legval(mu, [0.0] * l + [1.0])
    return B


class FrontNet(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer('B', torch.cat(
            [torch.randn(2, NF, generator=g) * s for s in SCALES], -1))     # (2,112)
        self.register_buffer('LB', torch.tensor(legendre_basis(), dtype=torch.float32))
        self.mode = nn.Parameter(torch.ones(LB_ORDER + 1))                  # 逐模态可学习倍数
        nin = 2 * self.B.shape[1] + NBLK
        self.body = nn.Sequential(
            nn.Linear(nin, WIDE), nn.GELU(),
            nn.Linear(WIDE, WIDE), nn.GELU(),
            nn.Linear(WIDE, WIDE), nn.GELU(),
            nn.Linear(WIDE, D_LAT))
        self.slin = nn.ModuleList([nn.Linear(D_LAT, D_LAT) for _ in range(NSCAT)])
        self.sln = nn.ModuleList([nn.LayerNorm(D_LAT) for _ in range(NSCAT)])
        self.out = nn.Linear(D_LAT, 1)

    def _A(self, g):
        """① 由 g 生成角向算子（行和 = 1；初值等于 HG 算子）。"""
        l = torch.arange(self.LB.shape[0], device=self.LB.device, dtype=self.LB.dtype)
        c = (float(g) ** l) * self.mode
        K = torch.einsum('l,lij->ij', c, self.LB)
        W = torch.tensor(WQ, device=K.device, dtype=K.dtype).view(1, -1)
        A = K * W
        return A / A.sum(1, keepdim=True).clamp_min(1e-30)

    def kernel_ops(self, gvals):
        u = torch.unique(gvals)
        mats = torch.stack([self._A(v) for v in u])                        # (U,KQ,KQ)
        idx = (gvals.view(-1, 1) == u.view(1, -1)).float().argmax(1)
        return mats[idx]

    def forward(self, blk, rel, gvals):
        q = 2.0 * float(np.pi) * torch.matmul(rel, self.B)
        if ZERO_ENC:
            z = torch.cat([torch.zeros_like(q), torch.zeros_like(q), blk], -1)
        else:
            z = torch.cat([torch.sin(q), torch.cos(q), blk], -1)
        h = self.body(z)
        A = self.kernel_ops(gvals)
        s = h
        for i in range(NSCAT - 1):
            s = self.sln[i](torch.tanh(self.slin[i](torch.matmul(A, s)))) + h
        out = h + self.sln[-1](torch.tanh(self.slin[-1](torch.matmul(A, s))))
        return self.out(out).squeeze(-1)


def build_batch(med, ai, ii, jj, kk):
    """构造 (blk, tgt, wgt, rel)；ai/ii/jj 等长（每个"条目×节点"一项），kk (N,KQ)。"""
    N = len(ai)
    a_e = np.repeat(ai, KQ); i_e = np.repeat(ii, KQ); j_e = np.repeat(jj, KQ)
    k_e = np.asarray(kk, np.int64).reshape(-1)
    sx = np.repeat(med.sxy[ai, 0], KQ); sy = np.repeat(med.sxy[ai, 1], KQ)
    rel = np.stack([i_e * H - sx, j_e * H - sy], -1).astype(np.float32)
    medv = med.med
    extra = np.empty((len(a_e), 5), np.float32)
    extra[:, 0] = medv[0] / SIG_MAX                      # σ_t/σ_t^max
    extra[:, 1] = medv[1] / medv[0]                      # ω_0 = σ_s/σ_t
    extra[:, 2] = medv[2]                                # g
    extra[:, 3] = sx; extra[:, 4] = sy
    kp_e = np.repeat(med.KP[ai], KQ)
    same = k_e == kp_e
    idp = np.asarray(med.ID[a_e, i_e, j_e], np.float64)
    lg = np.where(same, np.log(idp + 1e-12), LOG_EPS).astype(np.float32)
    if _KCACHE.get('g') != med.g:                       # ② 一次散射源先验
        _KCACHE['g'] = med.g
        _KCACHE['KH'] = np.asarray(hg_kernel(med.g), np.float64)
    KH = _KCACHE['KH']
    ss = np.log(medv[1] * WQ[kp_e] * KH[k_e, kp_e] * idp + 1e-30).astype(np.float32)
    blk = np.concatenate([(medv[0] * rel).astype(np.float32),
                          VC[k_e].astype(np.float32), VC[kp_e].astype(np.float32),
                          extra, lg[:, None]], -1)
    if NO_IDIR:
        blk[..., -1] = 0.0                      # 排除弹道项通道
    if ZERO_COLS:
        blk = _apply_zero_cols(blk)
    I = np.exp(np.asarray(med.Y[a_e, i_e, j_e, k_e], np.float64)) - 1e-12
    Id = np.where(same, idp, 0.0)
    Itgt = np.maximum(I.astype(np.float64), 0.0) if MODE_FRONT == 'full' \
        else np.maximum(I - Id, 0.0)
    tgt = np.log(Itgt + 1e-12).astype(np.float32)
    wgt = np.sqrt(Itgt / (Itgt + 1e-7)).astype(np.float32)
    return (blk.reshape(N, KQ, -1), tgt.reshape(N, KQ), wgt.reshape(N, KQ),
            rel.reshape(N, KQ, 2))


def phi_of_entries_fast(net, med, idx, chunk=96, rows_per_block=32768, return_I=False):
    """同 phi_of_entries 的结果，但把特征构建搬到显存、放大批次、只回传 Φ。

    当前实现的等价性说明：
      · 13 列特征与 Fourier 基都在 GPU 上广播拼出，CPU 只上传每条的小标量与 ID 场；
      · 每块的 Φ 直接在 GPU 上求积，只把 (n,NS,NS) 的 Φ 回传（每块约 0.3 MB）；
      · rows_per_block 放大，减少内核启动与同步次数。
    """
    NA = len(idx)
    Q = NS * NS
    out = np.empty((NA, NS, NS), np.float32)
    Ifield = np.empty((NA, NS, NS, KQ), np.float32) if return_I else None
    dev = DEV
    vc_t = torch.tensor(VC, dtype=torch.float32, device=dev)          # (KQ,2)
    wq_t = torch.tensor(WQ, dtype=torch.float32, device=dev)          # (KQ,)
    ar = torch.arange(NS, dtype=torch.float32, device=dev) * H
    kk = torch.arange(KQ, device=dev)
    KH = torch.tensor(hg_kernel(med.g), dtype=torch.float32, device=dev)   # (KQ,KQ)
    st, sm, gval = float(med.med[0]), float(med.med[1]), float(med.med[2])
    with torch.no_grad():
        for a0 in range(0, NA, chunk):
            sub = np.asarray(idx[a0:a0 + chunk], np.int64)
            n = len(sub)
            IDt = torch.as_tensor(np.asarray(med.ID[sub], np.float32), device=dev)   # (n,NS,NS)
            kp = torch.as_tensor(np.asarray(med.KP[sub], np.int64), device=dev)      # (n,)
            sxy = torch.as_tensor(np.asarray(med.sxy[sub], np.float32), device=dev)  # (n,2)
            sx, sy = sxy[:, 0], sxy[:, 1]
            relx = (ar[None, :, None] - sx[:, None, None]).expand(n, NS, NS)          # (n,NS,NS)
            rely = (ar[None, None, :] - sy[:, None, None]).expand(n, NS, NS)
            logID = torch.log(IDt + 1e-12)                                           # (n,NS,NS)
            sscof = (sm * wq_t[None, :] * KH.t()[kp, :])                             # (n,KQ)
            onehot = (kk[None, :] == kp[:, None])                                    # (n,KQ)
            blk = torch.empty((n, NS, NS, KQ, NBLK), dtype=torch.float32, device=dev)
            blk[..., 0] = (st * relx)[:, :, :, None]
            blk[..., 1] = (st * rely)[:, :, :, None]
            blk[..., 2] = vc_t[:, 0]
            blk[..., 3] = vc_t[:, 1]
            blk[..., 4] = vc_t[kp, 0][:, None, None, None]
            blk[..., 5] = vc_t[kp, 1][:, None, None, None]
            blk[..., 6] = st / SIG_MAX
            blk[..., 7] = sm / st
            blk[..., 8] = gval
            blk[..., 9] = sx[:, None, None, None]
            blk[..., 10] = sy[:, None, None, None]
            lgk = torch.where(onehot[:, None, None, :], logID[..., None],
                              torch.full_like(blk[..., 11], LOG_EPS))
            blk[..., 11] = 0.0 if NO_IDIR else lgk
            if ZERO_COLS:
                blk = _apply_zero_cols(blk)
            # [已移除] blk[..., 12] = 一次散射源先验
            blk4 = blk.reshape(n * Q, KQ, NBLK)
            rel4 = torch.stack([relx, rely], -1).reshape(n * Q, 1, 2).expand(n * Q, KQ, 2)
            del blk, lgk
            NR = n * Q
            gv = torch.full((min(rows_per_block, NR),), gval, device=dev)
            Io = torch.empty((NR, KQ), dtype=torch.float32, device=dev)
            for b in range(0, NR, rows_per_block):
                e = min(b + rows_per_block, NR)
                _o = net(blk4[b:e], rel4[b:e], gv[:e - b])
                if OUT_LIN:
                    Io[b:e] = _o.clamp_min(0.0)
                else:
                    Io[b:e] = torch.exp(_o.clamp(-40.0, 0.0)) - 1e-12
            Io.clamp_(min=0.0)
            Pnet = (Io * wq_t).sum(-1).reshape(n, NS, NS)
            if MODE_FRONT == 'full':
                out[a0:a0 + n] = Pnet.cpu().numpy()
            else:
                Phid = IDt * wq_t[kp][:, None, None]
                out[a0:a0 + n] = (Phid + Pnet).cpu().numpy()
            if return_I:
                Ifield[a0:a0 + n] = Io.reshape(n, NS, NS, KQ).cpu().numpy()
    return (out, Ifield) if return_I else out


def phi_of_entries(net, med, idx, chunk=48, rowblk=262144):
    """给定条目的整场通量 Φ (NA,41,41)：解析直接项 + 网络散射部分。

    向量化版：一个 chunk 内所有 (条目, 节点, 方向) 的特征用**广播 + 整块张量运算**一次拼好
    前向按 rowblk 行分块上卡，避免逐节点循环与重复 repeat/tile。
    """
    NA = len(idx)
    Q = NS * NS
    out = np.zeros((NA, NS, NS), np.float32)
    gv = torch.tensor(np.full(1, float(med.g)), device=DEV, dtype=torch.float32)
    medv = med.med
    KH = np.asarray(hg_kernel(med.g), np.float64)
    ii0 = np.repeat(np.arange(NS), NS)[None, :, None]          # (1,Q,1) i 慢 j 快
    jj0 = np.tile(np.arange(NS), NS)[None, :, None]
    kk0 = np.tile(np.arange(KQ), (Q, 1))[None, :, :]           # (1,Q,KQ)
    with torch.no_grad():
        for a0 in range(0, NA, chunk):
            sub = np.asarray(idx[a0:a0 + chunk], np.int64)
            n = len(sub)
            ai = sub[:, None, None]
            ii = np.broadcast_to(ii0, (n, Q, KQ))
            jj = np.broadcast_to(jj0, (n, Q, KQ))
            kk = np.broadcast_to(kk0, (n, Q, KQ))
            kp = med.KP[sub][:, None, None]                     # (n,1,1)
            sx = med.sxy[sub, 0][:, None, None]
            sy = med.sxy[sub, 1][:, None, None]
            relx = (ii * H - sx).astype(np.float32)             # (n,Q,KQ)
            rely = (jj * H - sy).astype(np.float32)
            same = (kk == kp)
            idp = np.asarray(med.ID[ai, ii, jj], np.float64)
            lg = np.where(same, np.log(idp + 1e-12), LOG_EPS).astype(np.float32)
            ss = np.log(medv[1] * WQ[kk] * KH[kk, kp] * idp + 1e-30).astype(np.float32)
            stc = np.float32(medv[0])
            blk = np.empty((n, Q, KQ, NBLK), np.float32)
            blk[..., 0] = stc * relx
            blk[..., 1] = stc * rely
            blk[..., 2] = VC[kk, 0]
            blk[..., 3] = VC[kk, 1]
            blk[..., 4] = VC[kp, 0]
            blk[..., 5] = VC[kp, 1]
            blk[..., 6] = medv[0] / SIG_MAX
            blk[..., 7] = medv[1] / medv[0]
            blk[..., 8] = medv[2]
            blk[..., 9] = sx
            blk[..., 10] = sy
            blk[..., 11] = 0.0 if NO_IDIR else lg
            if ZERO_COLS:
                blk = _apply_zero_cols(blk)
            # [已移除] blk[..., 12] = 一次散射源先验
            blk4 = blk.reshape(n * Q, KQ, NBLK)               # 每个 (条目,节点) 一行，KQ 方向为块
            rel4 = np.stack([relx, rely], -1).reshape(n * Q, KQ, 2)
            NR = blk4.shape[0]
            step = max(1, rowblk // KQ)
            o = np.empty((n, Q, KQ), np.float32)
            ov = o.reshape(NR, KQ)
            for b in range(0, NR, step):
                sl = slice(b, min(b + step, NR))
                pred = net(torch.tensor(blk4[sl], device=DEV),
                           torch.tensor(np.ascontiguousarray(rel4[sl]), device=DEV),
                           gv.expand(sl.stop - sl.start))
                ov[sl] = pred.cpu().numpy()
            if OUT_LIN:
                Io = np.maximum(o.reshape(n, NS, NS, KQ), 0.0)
            else:
                Io = np.exp(np.clip(o.reshape(n, NS, NS, KQ), -40.0, 0.0)) - 1e-12
            np.maximum(Io, 0.0, out=Io)
            Pnet = (Io * WQ[None, None, None, :]).sum(-1)
            if MODE_FRONT == 'full':
                out[a0:a0 + n] = Pnet.astype(np.float32)
            else:
                Phid = np.asarray(med.ID[sub], np.float64) * WQ[med.KP[sub]][:, None, None]
                out[a0:a0 + n] = (Phid + Pnet).astype(np.float32)
    return out
