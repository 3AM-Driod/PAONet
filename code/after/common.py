# -*- coding: utf-8 -*-
"""NetWork/after：基于注意力的任意几何 RTE 学习网络 —— 公共常量与路径。

学习目标（最终定稿）
    R = Φ_全场 − Φ_背景响应          纯背景响应残差（不含任何解析弹道项）
    角向形式上以对数比为目标：D = Y_全场 − Y_背景，Y = log(I + 1e-12)
    装配：I_θ = (I_背景 + EPS)·exp(D_θ) − EPS，Φ_θ = Σ_k w_k I_θ,k

采样（正式版）
    固定 N = 256 个 token，全部落在离散体素边界两侧 |d| ≤ ℓ_bg 的带内：
    密度 ρ(d) = 1 + 2·exp(−d²/2ℓ²) 给出"越靠近界面越密"的相对权重，
    先按 ρ 抽 8000 个候选，再用最远点采样降到 256 个点（等间距、无聚类、无方向偏好）。

特征（5 维，token 与查询共用编码器；无任何边界手工标签、无绝对位置、无先验场幅值）
    (s, ρ)/ℓ；σ_t/σ_t^max、ω0、g

偏置（每对 1 个标量，与场无关，故每脉冲算一次、全层复用）
    τ = σ_t^bg·|Δr|（唯一保留项；|Δr|/ℓ 与其恒等、log K_g 经消融确认可去）

正式版的构成（由 5.5 节的分项对照确定）
    保留：背景响应当作装配基线（学习目标 = 相对先验的对数比）与三项物理偏置
    放弃：先验场作为输入、按碰撞阶的层门控、绝对坐标特征——三者在对照中不改善精度
"""
from __future__ import annotations

import os
import sys

# 分配器配置：减少碎片、抑制"溢出到共享显存"（必须在首次 CUDA 分配之前设置）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import consts as C                                                   # noqa: E402
from config import AFTER_BG, AFTER_CACHE, AFTER_SCENES, WEIGHTS     # noqa: E402

ROOT = str(WEIGHTS.parent)

# ---------------------------------------------------------------- 常量
NS, KQ, H = C.NS, C.KQ, C.H
VC, WQ = np.asarray(C.VC, np.float64), np.asarray(C.WQ, np.float64)
GX = np.arange(NS) / (NS - 1.0)
INNER = np.zeros((NS, NS), bool)
INNER[1:-1, 1:-1] = True
INNER_IDX = np.flatnonzero(INNER.ravel())          # 内区节点的线性索引（用于损失/评估）
EPS = 1e-12                                        # 与数据集 Y=log(I+EPS) 一致

S_DIR = os.environ.get('S_DIR', str(AFTER_SCENES))
B_DIR = os.environ.get('B_DIR', str(AFTER_BG))
CACHE = os.environ.get('CACHE', str(AFTER_CACHE))
WEIGHT_DIR = os.path.join(str(WEIGHTS), 'runs', 'after')

# 采样：连续界面折线距离 + 密度加权池 + 最远点采样
#   1) 界面由占据节点提取为连续闭合折线，按折线到候选点的距离 d 定密度；
#   2) 密度 rho(d) = 1 + 2*exp(-d^2/(2*ell^2))，只保留 |d| <= TOK_BAND*ell_bg；
#   3) 按密度抽 TOK_POOL 个候选，再用最远点采样降到 NTOK 个。
NTOK = 256                                         # 固定 token 数
TOK_BAND = float(os.environ.get('TOK_BAND', '1.0'))     # 带宽（× ℓ_bg）
TOK_POOL = int(os.environ.get('TOK_POOL', '8000'))      # 候选池大小
TOK_GRID = int(os.environ.get('TOK_GRID', '161'))       # 候选网格边长
# static.npz（含 token 坐标）所在目录
STATIC_DIR = os.environ.get('STATIC_DIR', CACHE)

# 场景级划分（测试场景的全部脉冲均为测试）
SCENE_TEST_FRAC = 0.1
SCENE_SPLIT_SEED = 20261015

NFEAT = 5                                          # 逐点特征维数
NBIAS = 1                                          # 逐对偏置标量个数（只保留 τ_ij）
SIG_MAX = 16.0                                     # σ_t 的取值上界（数据集 σ_t ∈ [6,16]），用于归一化


def _flag(name):
    return os.environ.get(name, '0') == '1'


# 消融开关（默认全关；只把对应列置零，维数与参数量不变）
#   AB_NOGEOM ：A 组——射线几何 (s/ℓ, ρ/ℓ)
#   AB_NOMED  ：B 组——介质参数 (σ_t/σ_t^max, ω0, g)
#   AB_NOBIAS ：E1——去掉唯一的偏置项 τ_ij = σ_t^bg·|Δr|
AB = {
    'nobias': _flag('AB_NOBIAS'),
    'nogeom': _flag('AB_NOGEOM'),
    'nomed': _flag('AB_NOMED'),
}


def ab_name():
    """消融配置的短名（用于日志与结果表）。"""
    on = [k for k, v in AB.items() if v]
    return 'formal' if not on else 'no-' + '+'.join(on)


def hg_kernel(g):
    return np.asarray(C.hg_kernel(g), np.float64)


def pk_of(g):
    return hg_kernel(g) * WQ[None, :]


def new_run_dir(root=WEIGHT_DIR, prefix='train'):
    """新建 train1/train2/... 目录（与 NetWork/front 的做法一致）。"""
    os.makedirs(root, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(root, '%s%d' % (prefix, i))):
        i += 1
    d = os.path.join(root, '%s%d' % (prefix, i))
    os.makedirs(d)
    return d


def scene_ids():
    ids = [int(f[4:-4]) for f in os.listdir(S_DIR)
           if f.startswith('med_') and f.endswith('.npz')]
    have = {int(f[4:-4]) for f in os.listdir(B_DIR)
            if f.startswith('med_') and f.endswith('.npz')}
    return sorted(set(ids) & have)


def scene_split(ids=None, frac=SCENE_TEST_FRAC, seed=SCENE_SPLIT_SEED):
    """场景级划分：返回 (训练场景, 测试场景)。SMAX>0 时只取编号 < SMAX 的场景。"""
    ids = scene_ids() if ids is None else list(ids)
    smax = int(os.environ.get('SMAX', '0'))
    if smax > 0:
        ids = [i for i in ids if i < smax]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    nte = int(round(frac * len(ids)))
    te = sorted(ids[i] for i in perm[:nte])
    tr = sorted(ids[i] for i in perm[nte:])
    return tr, te


def load_pair(mid):
    a = np.load(os.path.join(S_DIR, 'med_%04d.npz' % mid), allow_pickle=False)
    b = np.load(os.path.join(B_DIR, 'med_%04d.npz' % mid), allow_pickle=False)
    return a, b


def ray_of(pulse):
    """脉冲 → (入流点 r0, 方向 om)。"""
    side, node, kdir = (int(v) for v in pulse)
    t = node / (NS - 1.0)
    r0 = np.array([0.0, t] if side == 0 else ([1.0, t] if side == 1 else
                                             ([t, 0.0] if side == 2 else [t, 1.0])))
    return r0, VC[kdir]


def ray_coords(P, r0, om):
    """点集 → (s 下游距离, rho 横向距离, 投影点)。"""
    d = np.atleast_2d(P) - r0
    s = d @ om
    perp = d - s[:, None] * om
    rho = np.linalg.norm(perp, axis=1)
    proj = r0[None, :] + s[:, None] * om[None, :]
    return s, rho, proj


def field_at(P, F):
    """在 41×41 节点场上双线性取值；P 为域坐标 (M,2)。"""
    q = np.clip(np.asarray(P, np.float64) * (NS - 1), 0.0, NS - 1 - 1e-9)
    i0 = q[:, 0].astype(int)
    j0 = q[:, 1].astype(int)
    fi, fj = q[:, 0] - i0, q[:, 1] - j0
    return ((1 - fi) * (1 - fj) * F[i0, j0] + fi * (1 - fj) * F[i0 + 1, j0]
            + (1 - fi) * fj * F[i0, j0 + 1] + fi * fj * F[i0 + 1, j0 + 1])


def node_index(P):
    """最近节点索引（分区常数介质读数的语义）。"""
    q = np.clip(np.round(np.asarray(P, np.float64) * (NS - 1)), 0, NS - 1)
    return q[:, 0].astype(int), q[:, 1].astype(int)
