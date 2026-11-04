# -*- coding: utf-8 -*-
"""把逐点特征与逐对物理偏置的构造搬到 GPU 上批量完成（正式版布局）。

输入只需要"小量"数据（每脉冲约 170 KB）：
    tokP/qryP (M,2) 点坐标、场景标量（σ_t^bg、g、ℓ）、
    每点的介质读数（log(σ_t/σ_t^bg)、ω0）、脉冲方向 om，以及目标 D/Yb/Phi。
输出：tok、qry（5 维特征）、bself (N,N,1)、bcross (Q,N,1)，全部在 GPU 上。

特征（5 维）：(s, ρ)/ℓ；σ_t/σ_t^max、ω0、g
偏置（1 项）：τ_ij = σ_t^bg|Δr|

正式版不含绝对坐标与背景先验场幅值两列特征（5.5 节的对照表明二者不改善精度），
也不含 Φ_bg 比值一项偏置。开关只用于物理偏置的分项消融。
"""
from __future__ import annotations

import torch

from common import AB, NBIAS, NFEAT, SIG_MAX


def build(s, dev):
    """s: 由 data.sample(..., slim=True) 给出的 CPU/GPU 小量字典 → 训练所需张量。"""
    def T(x, dt=torch.float32):
        return torch.as_tensor(x, dtype=dt, device=dev)

    tokP, qryP = T(s['tokP']), T(s['qryP'])
    tokR, qryR = T(s['tokR']), T(s['qryR'])              # (M,4) logσt/σt_bg, ω0, Δσt, Δω0
    r0, om, omu = T(s['r0']), T(s['om_raw']), T(s['omu'])     # (B,2)
    ell, st_bg, g = T(s['ell']), T(s['st_bg']), T(s['g'])     # (B,1)

    def feats(P, R):
        d = P - r0[:, None, :]
        sc = (d * omu[:, None, :]).sum(-1)
        rho = (d - sc[..., None] * omu[:, None, :]).norm(dim=-1)
        F = torch.empty(P.shape[0], P.shape[1], NFEAT, device=dev)
        _ratio = torch.exp(R[..., 0].clamp(-20.0, 20.0))       # σ_t/σ_t^bg（R 存的是对数比）
        F[..., 0] = sc / ell                        # 沿射线行程（以 ℓ 归一）
        F[..., 1] = rho / ell                       # 到射线轴的横向距离
        F[..., 2] = _ratio * st_bg / SIG_MAX        # σ_t/σ_t^max（以 σ_t 的上界归一）
        F[..., 3] = R[..., 1]                       # ω0
        F[..., 4] = g                               # g
        # 消融：把对应列置零（维数与参数量不变，只隔离信息）
        if AB['nogeom']:                            # A 组：射线几何
            F[..., 0:2] = 0.0
        if AB['nomed']:                             # B 组：介质参数
            F[..., 2:5] = 0.0
        return F

    tok = feats(tokP, tokR)
    qry = feats(qryP, qryR)

    def _bias_chunk(Pa_c, Pb_, om, st_bg, g, ell, dir_from_a, out, a0):
        """块内构造偏置并就地写入 out[:, a0:a0+Ac]；不物化单位向量 u。"""
        dx = Pb_[..., 0][:, None, :] - Pa_c[..., 0][:, :, None]      # (B,Ac,M)
        dy = Pb_[..., 1][:, None, :] - Pa_c[..., 1][:, :, None]
        r = (dx * dx + dy * dy).sqrt_()
        inv = 1.0 / r.clamp_min(1e-12)
        sgn = 1.0 if dir_from_a else -1.0
        mu = sgn * (dx * om[:, 0, None, None] + dy * om[:, 1, None, None]) * inv
        del dx, dy, inv
        o = out[:, a0:a0 + Pa_c.shape[1]]
        o[..., 0] = st_bg[:, :, None] * r                       # τ_ij（唯一偏置项）
        del r, mu

    def bias(Pa, Pb_, dir_from_a, chunk=192):
        A, M = Pa.shape[1], Pb_.shape[1]
        if AB['nobias']:                                            # 分项消融：整体置零
            return torch.zeros(Pa.shape[0], A, M, NBIAS, device=dev)
        out = torch.empty(Pa.shape[0], A, M, NBIAS, device=dev)
        for a0 in range(0, A, chunk):
            _bias_chunk(Pa[:, a0:a0 + chunk], Pb_, om, st_bg, g, ell,
                        dir_from_a, out, a0)
        return out

    bself = bias(tokP, tokP, True)
    bcross = bias(qryP, tokP, False)
    return dict(tok=tok, qry=qry, bself=bself, bcross=bcross,
                D=T(s['D']), Yb=T(s['Yb']), Phi=T(s['Phi']))
