# -*- coding: utf-8 -*-
"""NetWork/after 网络（正式版）：物理偏置注意力 + 逐点特征 + 24 方向输出头。

结构（与 5.5 节分项对照确定的正式版一致）
    1. 特征编码：token 与查询共用同一个 MLP（二者特征 schema 相同，NFEAT=9）
    2. L 层自注意力：全连接（N=256 很小），logits + 偏置 B_self(N,N,nhead)；
       各层等权（按碰撞阶的 ω0 门控已由分项对照确定去除）
    3. 1–2 层交叉注意力：查询对 token，logits + 偏置 B_cross(Q,N,nhead)
    4. 输出头：每查询 KQ=24 个对数比残差 D_θ
    5. 装配：I_θ = (I_背景 + EPS)·exp(D_θ) − EPS，Φ_θ = Σ_k w_k I_θ,k
       学习目标就是纯背景响应残差（不含任何解析弹道项）
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from common import EPS, KQ, NBIAS, NFEAT, WQ

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WQ_T = torch.tensor(WQ, dtype=torch.float32)
NHEADS = int(os.environ.get('NHEAD', '4'))   # 注意力头数（容量扫描用）


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, nhid=None, act=nn.SiLU):
        super().__init__()
        nhid = nout if nhid is None else nhid
        layers, d = [], nin
        for _ in range(nlayer - 1):
            layers += [nn.Linear(d, nhid), act()]
            d = nhid
        layers += [nn.Linear(d, nout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BiasNet(nn.Module):
    """1 个物理标量（τ_ij）→ 每头偏置（隐层取小值，避免 (B,A,M,nhid) 的显存膨胀）。"""

    def __init__(self, nhead, nhid=8):
        super().__init__()
        self.mlp = MLP(NBIAS, nhead, nlayer=2, nhid=nhid)
        self.gamma = nn.Parameter(torch.tensor(0.5))       # 碰撞衰减先验的幅度

    def forward(self, B):
        """B: (..., NBIAS=1) → (..., nhead)。"""
        base = -self.gamma * B[..., 0]
        return base[..., None] + self.mlp(B)


class SelfBlock(nn.Module):
    def __init__(self, F, nhead, drop=0.0):
        super().__init__()
        self.nhead = nhead
        self.ln1 = nn.LayerNorm(F)
        self.qkv = nn.Linear(F, 3 * F)
        self.proj = nn.Linear(F, F)
        self.ln2 = nn.LayerNorm(F)
        self.ffn = MLP(F, F, nlayer=2, nhid=2 * F)

    def forward(self, h, bias):
        B, N, F = h.shape
        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.nhead, F // self.nhead).transpose(1, 2)
        k = k.view(B, N, self.nhead, F // self.nhead).transpose(1, 2)
        v = v.view(B, N, self.nhead, F // self.nhead).transpose(1, 2)
        # 融合核（SDPA：优先 Flash / memory-efficient 后端），避免显式物化 (B,H,N,N) 分数矩阵
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        o = o.transpose(1, 2).reshape(B, N, F)
        h = h + self.proj(o)
        h = h + self.ffn(self.ln2(h))
        return h


class CrossBlock(nn.Module):
    def __init__(self, F, nhead):
        super().__init__()
        self.nhead = nhead
        self.lnq = nn.LayerNorm(F)
        self.lnk = nn.LayerNorm(F)
        self.wq = nn.Linear(F, F)
        self.wk = nn.Linear(F, F)
        self.wv = nn.Linear(F, F)
        self.proj = nn.Linear(F, F)
        self.ln2 = nn.LayerNorm(F)
        self.ffn = MLP(F, F, nlayer=2, nhid=2 * F)

    def forward(self, hq, ht, bias):
        Bq, Q, F = hq.shape
        N = ht.shape[1]
        q = self.wq(self.lnq(hq)).view(Bq, Q, self.nhead, F // self.nhead).transpose(1, 2)
        kk = self.wk(self.lnk(ht)).view(Bq, N, self.nhead, F // self.nhead).transpose(1, 2)
        v = self.wv(ht).view(Bq, N, self.nhead, F // self.nhead).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(q, kk, v, attn_mask=bias)
        o = o.transpose(1, 2).reshape(Bq, Q, F)
        hq = hq + self.proj(o)
        hq = hq + self.ffn(self.ln2(hq))
        return hq


class AfterNet(nn.Module):
    """输入：token/查询特征与两套偏置。输出：24 方向对数比残差。"""

    def __init__(self, F=192, nhead=NHEADS, nlayer=6, ncross=2):
        super().__init__()
        self.enc = MLP(NFEAT, F, nlayer=3, nhid=F)
        self.bias_self = BiasNet(nhead)
        self.bias_cross = BiasNet(nhead)
        self.blocks = nn.ModuleList([SelfBlock(F, nhead) for _ in range(nlayer)])
        self.cross = nn.ModuleList([CrossBlock(F, nhead) for _ in range(ncross)])
        self.head = MLP(F, KQ, nlayer=2, nhid=F)
        self.nlayer = nlayer
        # 输出头与偏置 MLP 的末层零初始化：训练从"零残差 = 背景响应"这一物理基线出发
        for m in (self.head, self.bias_self.mlp, self.bias_cross.mlp):
            last = [l for l in m.net if isinstance(l, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, tok, qry, bself, bcross):
        """tok (B,N,NFEAT) qry (B,Q,NFEAT) bself (B,N,N,NBIAS) bcross (B,Q,N,NBIAS)
        → D (B,Q,KQ)。"""
        ht = self.enc(tok)
        hq = self.enc(qry)
        Bs = self.bias_self(bself).permute(0, 3, 1, 2)      # (B, head, N, N)
        Bc = self.bias_cross(bcross).permute(0, 3, 1, 2)    # (B, head, Q, N)
        for blk in self.blocks:
            ht = blk(ht, Bs)
        for cr in self.cross:
            hq = cr(hq, ht, Bc)
        return self.head(hq)


MODE_AFTER = os.environ.get('MODE_AFTER', 'resid')   # resid|full|lin|logmag


def assemble(D, Yb):
    """按 MODE_AFTER 解释网络输出并合成完整场。

    resid ：输出为对数比，I = (I_bg+eps)*exp(D) - eps
    full  ：输出为完整场的对数，I = exp(D)
    lin   ：输出为扰动本身，I = I_bg + D
    logmag：输出为扰动幅度的对数，I = I_bg + max(0, exp(D) - eps)
    """
    if MODE_AFTER == 'full':
        I = torch.exp(D.clamp(min=-40.0, max=60.0))
    elif MODE_AFTER == 'lin':
        Ib = torch.exp(Yb.clamp(max=60.0)) - EPS
        I = Ib + D
    elif MODE_AFTER == 'logmag':
        Ib = torch.exp(Yb.clamp(max=60.0)) - EPS
        I = Ib + (torch.exp(D.clamp(min=-40.0, max=60.0)) - EPS).clamp_min(0.0)
    else:
        Ib = torch.exp(Yb) - EPS
        I = (Ib + EPS) * torch.exp(D) - EPS
    return I, (I * WQ_T.to(I.device)).sum(dim=-1)
