# -*- coding: utf-8 -*-
"""NetWork/after 数据管线：介质读数、界面侧重采样、逐点特征与物理偏置。

介质读数（与求解器一致，绝不插值）
    点 → 最近节点（Voronoi 语义）→ 该节点的 (σ_t, σ_s)；g 取场景级共用值
    （数据集生成时求解器只接受单个相函数核，g 为场景级共用，故读数与求解一致）
    场景既有离散体素掩膜（数据集 OCC），也支持由 GEO 解析形体 + 泛洪填洞重建
    （推理新几何时走这条，协议与生成器相同）

采样（只保留界面侧重一种）
    界面点：由占据节点提取连续闭合轮廓折线，按折线到候选点的距离采样
    界面壳：以连续折线为基准，在 |d| ≤ band·ℓ_bg 的带内按密度 ρ(d) 抽样
    总数固定为 NTOK（先按 ρ 抽池，再用最远点采样降到 NTOK；无网格步长依赖）

特征（9 维，token 与查询共用同一 schema；不含任何边界手工标签、无绝对位置、无先验场幅值）
    (s,ρ)/ℓ；log(σ_t/σ_t^bg)、ω0、g、Δσ_t/σ_t^bg、Δω0；τ_ray(s)、e^{−τ_ray}

偏置（每对 3 个标量）
    τ = σ_t^bg|Δr|；log K_g(Ω·u)；|Δr|/ℓ

样本
    tok (N,9)、qry (Q,9)、bself (N,N,3)、bcross (Q,N,3)、
    D (Q,KQ) 对数比目标 Y_全场−Y_背景、Yb (Q,KQ) 背景角向场、Phi (Q,) 真值 Φ
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.path import Path
from skimage import measure

from common import (B_DIR, CACHE, EPS, GX, KQ, NBIAS, NFEAT, NS, NTOK, S_DIR,
                    STATIC_DIR, load_pair, node_index, ray_coords, ray_of,
                    scene_split)


# ------------------------------------------------------------------ 介质
class Medium:
    """离散体素介质（41×41 节点）。"""

    def __init__(self, occ, p_bg, p_obj, g_scene):
        self.occ = np.asarray(occ, bool).copy()
        self.p_bg = np.asarray(p_bg, np.float64)
        self.p_obj = np.asarray(p_obj, np.float64)
        self.g = float(g_scene)
        self.st_bg, self.w0_bg = float(self.p_bg[0]), float(self.p_bg[1])
        self.st_obj, self.w0_obj = float(self.p_obj[0]), float(self.p_obj[1])
        self.ell_bg = 1.0 / max(self.st_bg, 1e-9)
        self.ell_obj = 1.0 / max(self.st_obj, 1e-9)
        self.mut = np.where(self.occ, self.st_obj, self.st_bg)
        self.mus = np.where(self.occ, self.st_obj * self.w0_obj, self.st_bg * self.w0_bg)

    @classmethod
    def from_file(cls, a):
        return cls(a['OCC'], a['P_BG'], a['P_OBJ'], float(a['P_BG'][2]))

    @staticmethod
    def occupancy_from_geo(geo_json, cell=NS):
        """由 GEO 解析形体在 (cell×cell) 网格上求占用，并按生成器协议泛洪填洞。"""
        import json

        from scipy import ndimage
        shapes = json.loads(geo_json)
        g = np.arange(cell) / (cell - 1.0)
        X, Y = np.meshgrid(g, g, indexing='ij')
        CN = {'circle': 96, 'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6}
        occ = np.zeros((cell, cell), bool)
        for s in shapes:
            dx, dy = X - s['cx'], Y - s['cy']
            ca, sa = np.cos(s['rot']), np.sin(s['rot'])
            u = dx * ca + dy * sa
            v = (-dx * sa + dy * ca) / s['asp']
            th = np.arctan2(v, u)
            aa = 2.0 * np.pi / CN[s['kind']]
            r0 = s['R'] * np.cos(0.5 * aa) / np.cos(np.mod(th, aa) - 0.5 * aa)
            r = r0 * (1.0 + sum(ak * np.sin(k * th + ph)
                                for k, ak, ph in zip(s['ks'], s['ak'], s['ph'])))
            occ |= np.hypot(u, v) <= np.maximum(r, 0.2 * s['R'])
        lab, _n = ndimage.label(~occ)
        border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
        border.discard(0)
        if border:
            occ |= (~occ & ~np.isin(lab, list(border)))
        return occ

    @classmethod
    def from_geo(cls, geo_json, p_bg, p_obj, g_scene):
        return cls(cls.occupancy_from_geo(geo_json), p_bg, p_obj, g_scene)

    def read(self, P):
        """点集 → (σ_t, σ_s, σ_t 反差, ω0 反差)（分区常数，最近节点）。"""
        i, j = node_index(P)
        st = self.mut[i, j]
        ss = self.mus[i, j]
        return st, ss, st / self.st_bg - 1.0, ss / np.maximum(st, 1e-12) - self.w0_bg


# ------------------------------------------------------------------ 体素边界与采样
def _boundary_loops(occ):
    """占据节点掩膜 → 连续闭合界面折线（节点坐标，非半格中点）。"""
    cs = measure.find_contours(occ.astype(np.float32), 1.0 - 1e-9,
                               fully_connected='high')
    n = occ.shape[0]
    loops = []
    for c in cs:
        if len(c) < 4:
            continue
        p = np.column_stack([c[:, 1], c[:, 0]]) / (n - 1.0)
        if np.linalg.norm(p[0] - p[-1]) > 1e-9:
            p = np.vstack([p, p[0]])
        loops.append(p)
    return loops


def _dist_loops(P, loops):
    """点集到全部界面折线的无符号距离。"""
    P = np.atleast_2d(np.asarray(P, np.float64))
    best = np.full(len(P), np.inf)
    for loop in loops:
        A = loop[:-1]
        B = loop[1:]
        V = B - A
        L2 = np.maximum(np.sum(V ** 2, axis=1), 1e-30)
        for a, v, l2 in zip(A, V, L2):
            t = np.clip(((P - a) @ v) / l2, 0.0, 1.0)
            q = a + t[:, None] * v
            d = np.linalg.norm(P - q, axis=1)
            best = np.minimum(best, d)
    return best


def _inside_loops(P, loops):
    """点集是否位于界面闭合折线内部（多轮廓用奇偶规则）。"""
    P = np.atleast_2d(np.asarray(P, np.float64))
    m = np.zeros(len(P), bool)
    for loop in loops:
        m ^= Path(loop).contains_points(P)
    return m


def scene_tokens(med, nt=NTOK, rng=None, band=None, npool=None, ngrid=None):
    """界面邻近重采样点（正式版）→ (P(N,2), kind(N,), parent(N,))；kind 仅诊断用。

    1. 界面 = 占据节点提取的连续闭合折线（含角点），距离 d 按折线度量；
    2. 候选 = ngrid×ngrid 的格心网格，密度 ρ(d) = 1+2exp(−d²/2ℓ²)（ℓ = ℓ_bg），
       只保留 |d| ≤ band·ℓ 的带内候选（带宽默认 1 个背景平均自由程）；
    3. 按 ρ 独立抽 npool 个候选，再以最远点采样降到 nt 个（等间距、无聚类、无方向偏好）。

    注：三种位置（界面附近、带内两侧）在特征与网络处理上完全一致，无边界标签。
    """
    from common import TOK_BAND, TOK_GRID, TOK_POOL
    band = TOK_BAND if band is None else band
    npool = TOK_POOL if npool is None else npool
    ngrid = TOK_GRID if ngrid is None else ngrid
    rg = np.random.default_rng(0 if rng is None else int(rng))
    ell = float(med.ell_bg)
    loops = _boundary_loops(med.occ)
    if not loops:                                    # 无界面（极端情况）：退回均匀撒点
        P = rg.random((nt, 2))
        return P, np.zeros(nt, int), np.arange(nt)
    g = (np.arange(ngrid) + 0.5) / ngrid
    CAND = np.column_stack([np.repeat(g, ngrid), np.tile(g, ngrid)])
    d = _dist_loops(CAND, loops)
    sd = np.where(_inside_loops(CAND, loops), -d, d)
    u = np.abs(sd) / max(ell, 1e-9)
    w = (1.0 + 2.0 * np.exp(-u ** 2 / 2.0)) * (u <= band)
    if w.sum() <= 0:
        w = np.ones_like(w)
    pool = CAND[rg.choice(len(CAND), size=min(npool, len(CAND)), p=w / w.sum())]
    u_pool = _dist_loops(pool, loops) / max(ell, 1e-9)
    k0 = int(np.argmax(np.exp(-(u_pool ** 2) / 2.0)))    # 最靠近界面的候选作起点
    sel = [k0]
    dmin = np.linalg.norm(pool - pool[k0], axis=1)
    for _ in range(nt - 1):
        k = int(np.argmax(dmin))
        sel.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(pool - pool[k], axis=1))
    P = pool[np.asarray(sel)]
    return P, np.zeros(len(P), int), np.arange(len(P))


# ------------------------------------------------------------------ 逐脉冲量
def tau_ray_table(med, r0, omu, n=192):
    """沿入流射线的累计光学厚度表（用**真实介质**）；omu 为单位方向。"""
    t = np.linspace(0.0, 1.0, n)[:, None]
    P = r0[None, :] + t * omu[None, :]
    on = (P[:, 0] >= 0.0) & (P[:, 0] <= 1.0) & (P[:, 1] >= 0.0) & (P[:, 1] <= 1.0)
    i, j = node_index(P)
    st = np.where(on, med.mut[i, j], 0.0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (st[1:] + st[:-1]) * seg)])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, tau


def tau_ray_at(P, r0, omu, s_tab, tau_tab):
    """点集 → 其射线投影处的累计光学厚度（表插值），以及 (s, ρ)。"""
    s, rho, _proj = ray_coords(P, r0, omu)
    tau = np.interp(np.clip(s, s_tab[0], s_tab[-1]), s_tab, tau_tab)
    return np.where(s > 0.0, tau, 0.0), s, rho


def features(P, med, r0, omu):
    """5 维特征（token 与查询共用；与 feat_torch 的布局逐列一致）。"""
    from common import SIG_MAX
    st, ss, _d_st, _d_w0 = med.read(P)
    w0 = ss / np.maximum(st, 1e-12)
    s, rho, _proj = ray_coords(P, r0, omu)
    ell = med.ell_bg
    F = np.empty((len(P), NFEAT), np.float32)
    F[:, 0] = s / ell
    F[:, 1] = rho / ell
    F[:, 2] = st / SIG_MAX
    F[:, 3] = w0
    F[:, 4] = med.g
    return F


def bias_scalars(Pa, Pb, med, om_raw, dir_from_a=False):
    """1 个偏置标量 τ_ij = σ_t^bg·|Δr|（其余两项经消融确认可去）。"""
    d = np.asarray(Pb, np.float64)[None, :, :] - np.asarray(Pa, np.float64)[:, None, :]
    r = np.linalg.norm(d, axis=-1)
    B = np.empty(r.shape + (NBIAS,), np.float32)
    B[..., 0] = med.st_bg * r
    return B


def scene_static(med, mid, nt=NTOK):
    """每场景与脉冲无关的静态部分：界面侧重采样点与各点介质读数。

    （预存到缓存里，避免每个 DataLoader worker 重复计算。）
    """
    P, kind, parent = scene_tokens(med, nt, rng=mid)
    Q = np.column_stack([np.repeat(GX, NS), np.tile(GX, NS)])

    def readings(X):
        st, ss, d_st, d_w0 = med.read(X)
        w0 = ss / np.maximum(st, 1e-12)
        return np.column_stack([np.log(np.maximum(st, 1e-12) / med.st_bg),
                                w0, d_st, d_w0]).astype(np.float32)

    return dict(tokP=P.astype(np.float32), tokR=readings(P),
                qryP=Q.astype(np.float32), qryR=readings(Q))


# ------------------------------------------------------------------ 数据集
class AfterData:
    """场景级划分 + 场景缓存 + 按需生成样本。

    优先使用 prep_cache.py 生成的未压缩记忆映射缓存（按脉冲访问是廉价切片）；
    没有缓存时回退为直接读压缩 npz。
    """

    def __init__(self, ntok=NTOK, verbose=True, cache=8):
        self.ntok = ntok
        self.tr, self.te = scene_split()
        self._cache = {}
        self._order = []
        self._max = cache
        self._np = {}
        if verbose:
            print('场景划分：训练 %d 个、测试 %d 个（测试场景的全部脉冲只作测试）'
                  % (len(self.tr), len(self.te)), flush=True)

    def n_pulse(self, mid):
        """该场景的脉冲条数（只读很小的 IDX，可缓存；供 DataLoader 建索引）。"""
        if mid not in self._np:
            cd = os.path.join(CACHE, '%04d' % mid)
            f_meta = os.path.join(cd, 'meta.npz')
            if os.path.exists(f_meta):
                self._np[mid] = int(len(np.load(f_meta, allow_pickle=False)['IDX']))
            else:
                a, _b = load_pair(mid)
                self._np[mid] = int(len(np.asarray(a['IDX'])))
        return self._np[mid]

    def _scene(self, mid):
        if mid in self._cache:
            self._order.remove(mid)
            self._order.append(mid)
            return self._cache[mid]
        if len(self._order) >= self._max:
            self._cache.pop(self._order.pop(0), None)
        cd = os.path.join(CACHE, '%04d' % mid)
        st_file = os.path.join(STATIC_DIR, '%04d' % mid, 'static.npz')
        if not os.path.exists(st_file):              # 换采样方案时静态部分另存目录
            st_file = os.path.join(cd, 'static.npz')
        if os.path.exists(os.path.join(cd, 'D.npy')):
            m = np.load(os.path.join(cd, 'meta.npz'), allow_pickle=False)
            arr = dict(D=np.load(os.path.join(cd, 'D.npy'), mmap_mode='r'),
                       Yb=np.load(os.path.join(cd, 'Yb.npy'), mmap_mode='r'),
                       PHIb=np.load(os.path.join(cd, 'Pb.npy'), mmap_mode='r'),
                       PHIf=np.load(os.path.join(cd, 'Pf.npy'), mmap_mode='r'))
            med = Medium.from_file(m)
            head = dict(IDX=np.asarray(m['IDX']).copy(), src='cache')
        else:
            a, b = load_pair(mid)
            arr = dict(D=(a['Y'].astype(np.float32) - b['Y'].astype(np.float32)),
                       Yb=b['Y'].astype(np.float32),
                       PHIb=b['PHI'].astype(np.float32),
                       PHIf=a['PHI'].astype(np.float32))
            med = Medium.from_file(a)
            head = dict(IDX=np.asarray(a['IDX']).copy(), src='npz')
        if os.path.exists(st_file):                     # 静态部分已预存：单次小文件读取
            z = np.load(st_file, allow_pickle=False)
            stt = {k: np.asarray(z[k]) for k in ('tokP', 'tokR', 'qryP', 'qryR')}
        else:
            stt = scene_static(med, mid, self.ntok)
        self._cache[mid] = dict(med=med, **head, **arr, **stt)
        self._order.append(mid)
        return self._cache[mid]

    def sample(self, mid, i, slim=True):
        """一个 (场景, 脉冲) 的样本：返回"小量"数据（坐标、读数、射线表、先验场、目标），
        逐点特征与偏置由 feat_torch.build 在 GPU 上批量算。

        slim 参数保留只为兼容既有调用；逐点特征的手工路径已随特征改版删除。
        """
        s = self._scene(mid)
        med = s['med']
        pulse = np.asarray(s['IDX'], int)[i]
        r0, om_raw = ray_of(pulse)
        omu = om_raw / max(float(np.linalg.norm(om_raw)), 1e-12)
        s_tab, tau_tab = tau_ray_table(med, r0, omu)
        pb = np.asarray(s['PHIb'][i], np.float64)
        pk = max(float(pb.max()), 1e-30)
        D = np.array(s['D'][i], np.float32).reshape(-1, KQ)
        Yb = np.array(s['Yb'][i], np.float32).reshape(-1, KQ)
        Phi = np.array(s['PHIf'][i], np.float32).reshape(-1)
        return dict(tokP=s['tokP'], tokR=s['tokR'], qryP=s['qryP'], qryR=s['qryR'],
                    r0=r0, om_raw=om_raw, omu=omu, s_tab=s_tab, tau_tab=tau_tab,
                    Pb=np.asarray(pb, np.float32),
                    pk=np.array([pk], np.float32),
                    ell=np.array([med.ell_bg], np.float32),
                    st_bg=np.array([med.st_bg], np.float32),
                    g=np.array([med.g], np.float32),
                    w0=np.array([med.w0_bg], np.float32),
                    D=D, Yb=Yb, Phi=Phi, mid=mid, i=i)
