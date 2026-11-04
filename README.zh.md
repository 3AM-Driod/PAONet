# PAONet

PAONet 是论文所述两段式 RTE 求解构架的参考实现：先用背景响应网络求解均匀参考介质，再用扰动响应网络修正材料界面附近的场。

## 仓库结构

```text
PAONet/
├─ code/
│  ├─ config.py                  # 路径配置
│  ├─ consts.py                  # 41×41 网格、24 方向、HG 核
│  ├─ direct_term.py             # 解析弹道项
│  ├─ front/
│  │  ├─ front_net.py            # 背景响应网络
│  │  ├─ train_front.py
│  │  ├─ eval_front_arm.py       # 逐脉冲评估
│  │  ├─ eval_front_scene.py     # 逐场景评估
│  │  ├─ data.py
│  │  └─ common.py
│  ├─ after/
│  │  ├─ after_net.py            # 扰动响应网络
│  │  ├─ train_after.py
│  │  ├─ eval_after_arm.py       # 逐脉冲评估
│  │  ├─ eval_after_scene.py     # 逐场景评估
│  │  ├─ prep_cache.py           # 由 scenes/bg 生成完整缓存
│  │  ├─ prep_static.py          # 只重建 static.npz
│  │  ├─ data.py
│  │  ├─ feat_torch.py
│  │  ├─ loader.py
│  │  └─ common.py
│  ├─ eval_chain.py              # 前段 + 后段端到端评估
│  └─ eval_ablations.py          # 批量评估消融权重
├─ weights/
│  ├─ formal/                    # 正式版前段/后段权重
│  ├─ ablations/                 # 表 5.6/5.7/5.8/5.10/5.11/5.15 权重
│  └─ README.md
├─ data/
│  ├─ front/data1/
│  ├─ after/data1/{scenes,bg}/
│  ├─ after/cache1/
│  └─ README.md
├─ scripts/                      # PowerShell 入口
├─ results/                      # 评估输出
└─ requirements.txt
```

## 安装

- Python 3.9
- PyTorch 2.x（建议 CUDA 版）
- numpy、scipy、matplotlib、scikit-image

```powershell
pip install -r requirements.txt
```

## 快速开始

```powershell
cd E:\RTE_JEPA\PAONet

# 自带子集上的端到端评估
python code\eval_chain.py

# 前段 / 后段单独评估
python code\front\eval_front_arm.py --ckpt weights\formal\front\front_net.pt `
       --data data\front\data1 --tag formal_front
python code\after\eval_after_arm.py --ckpt weights\formal\after\after_net.pt --tag formal_after

# 消融评估
python code\eval_ablations.py --kind after
python code\eval_ablations.py --kind front
python code\eval_ablations.py --kind all
```

结果写入 `results/`。

## 脚本入口

| 脚本 | 作用 |
|---|---|
| `scripts/train_front.ps1` | 训练背景响应网络 |
| `scripts/train_after.ps1` | 训练扰动响应网络 |
| `scripts/eval_front.ps1` | 评估前段权重 |
| `scripts/eval_after.ps1` | 评估后段权重 |
| `scripts/eval_chain.ps1` | 端到端评估 |
| `scripts/prep_cache.ps1` | 生成完整后段缓存 |
| `scripts/prep_static.ps1` | 只重建 `static.npz` |
| `scripts/eval_ablations.ps1` | 批量评估消融权重 |

所有脚本都是 Python 入口的薄封装，环境变量原样传递。

## 数据与缓存

自带数据包含 10 个场景；完整数据说明见 `data/README.md`。

由 `scenes/` 与 `bg/` 生成后段缓存：

```powershell
python code\after\prep_cache.py
```

只重建已有缓存的界面采样部分：

```powershell
python code\after\prep_static.py
```

可选环境变量：

| 变量 | 含义 |
|---|---|
| `CACHE` | 缓存根目录，默认 `data/after/cache1` |
| `S_DIR` | 场景目录，默认 `data/after/data1/scenes` |
| `B_DIR` | 背景目录，默认 `data/after/data1/bg` |
| `MEDIA` | 场景范围，例如 `0,20` 表示 `0..19` |
| `FORCE` | `1` 覆盖已有文件 |

每个缓存条目包含：

```text
D.npy      对数比目标
Yb.npy     背景角向场
Pb.npy     背景积分通量
Pf.npy     参考积分通量
meta.npz   IDX/OCC/P_BG/P_OBJ/GEO/TE
static.npz token 坐标与介质读数
```

缓存脚本只读取已生成的场景文件，不调用参考求解器。

## 训练

前段：

```powershell
python code\front\train_front.py
```

后段：

```powershell
python code\after\train_after.py
```

常用环境变量：

| 模块 | 变量 | 默认值 |
|---|---|---|
| front | `NSTEP` | 100000 |
| front | `LR_FRONT` | 2e-3 |
| front | `FW`、`FL` | 384、64 |
| front | `NSC_FRONT` | 2 |
| after | `F` | 192 |
| after | `NLAYER` | 6 |
| after | `NCROSS` | 2 |
| after | `NSTEP` | 15000 |
| after | `TPER` | 20000 |
| after | `BATCH` | 64 |
| after | `NWORKERS` | 6 |

新训练结果写入 `weights/runs/`，不会覆盖 `formal/` 与 `ablations/`。

## 评估口径

- 前段：`eval_front_arm.py`，逐脉冲内区相对 L2；
- 后段：`eval_after_arm.py`，以参考背景场为输入，评估扰动修正；
- 端到端：`eval_chain.py`，前段 + 后段，同时给出逐脉冲与逐场景 RMSPE；
- 消融：`eval_ablations.py`，按权重自动设置输出模式和消融开关，汇总到 `results/ablation_summary.json`。

完整测试集上正式版逐场景结果为 `1.0445%`。

## 消融权重

`weights/ablations/` 按正文表格组织：

| 目录 | 内容 |
|---|---|
| `tab5_6_loss/front/L_bg1`–`L_bg4` | 前段损失函数变体 |
| `tab5_6_loss/after/L_pt1`–`L_pt5` | 后段损失函数变体 |
| `tab5_7_prior/front/{with_prior,no_prior}` | 解析先验消融 |
| `tab5_8_structure/single_stage` | 单段式网络 |
| `tab5_10_feat_front/front/*` | 前段输入特征消融 |
| `tab5_11_feat_after/after/*` | 后段输入特征消融 |
| `tab5_15_capacity/{3M,1M,300K,30K}` | 四档容量 |
