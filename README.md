# PAONet

PAONet is the reference implementation of the two-stage RTE solver described in the paper:
a background-response network for a homogeneous reference medium, followed by a
perturbation-response network that corrects the field around material interfaces.

## Repository layout

```text
PAONet/
├─ code/
│  ├─ config.py                  # path configuration
│  ├─ consts.py                  # 41×41 grid, 24 directions, HG kernel
│  ├─ direct_term.py             # analytic ballistic term
│  ├─ front/
│  │  ├─ front_net.py            # background-response network
│  │  ├─ train_front.py
│  │  ├─ eval_front_arm.py       # per-pulse evaluation
│  │  ├─ eval_front_scene.py     # per-scene evaluation
│  │  ├─ data.py
│  │  └─ common.py
│  ├─ after/
│  │  ├─ after_net.py            # perturbation-response network
│  │  ├─ train_after.py
│  │  ├─ eval_after_arm.py       # per-pulse evaluation
│  │  ├─ eval_after_scene.py     # per-scene evaluation
│  │  ├─ prep_cache.py           # build complete cache from scenes/bg
│  │  ├─ prep_static.py          # rebuild only static.npz
│  │  ├─ data.py
│  │  ├─ feat_torch.py
│  │  ├─ loader.py
│  │  └─ common.py
│  ├─ eval_chain.py              # end-to-end front + after evaluation
│  └─ eval_ablations.py          # evaluate all ablation checkpoints
├─ weights/
│  ├─ formal/                    # formal front/after weights
│  ├─ ablations/                 # table 5.6/5.7/5.8/5.10/5.11/5.15 checkpoints
│  └─ README.md
├─ data/
│  ├─ front/data1/
│  ├─ after/data1/{scenes,bg}/
│  ├─ after/cache1/
│  └─ README.md
├─ scripts/                      # PowerShell entry points
├─ results/                      # evaluation outputs
└─ requirements.txt
```

## Installation

- Python 3.9
- PyTorch 2.x (CUDA build recommended)
- numpy, scipy, matplotlib, scikit-image

```powershell
pip install -r requirements.txt
```

## Quick start

```powershell
cd E:\RTE_JEPA\PAONet

# End-to-end evaluation on the bundled subset
python code\eval_chain.py

# Arm-level evaluation
python code\front\eval_front_arm.py --ckpt weights\formal\front\front_net.pt `
       --data data\front\data1 --tag formal_front
python code\after\eval_after_arm.py --ckpt weights\formal\after\after_net.pt --tag formal_after

# Ablation evaluation
python code\eval_ablations.py --kind after
python code\eval_ablations.py --kind front
python code\eval_ablations.py --kind all
```

Results are written to `results/`.

## Script entry points

| Script | Purpose |
|---|---|
| `scripts/train_front.ps1` | train the background-response network |
| `scripts/train_after.ps1` | train the perturbation-response network |
| `scripts/eval_front.ps1` | evaluate a front checkpoint |
| `scripts/eval_after.ps1` | evaluate an after checkpoint |
| `scripts/eval_chain.ps1` | end-to-end evaluation |
| `scripts/prep_cache.ps1` | build the complete after-network cache |
| `scripts/prep_static.ps1` | rebuild only `static.npz` |
| `scripts/eval_ablations.ps1` | evaluate ablation checkpoints |

All scripts are thin wrappers around the Python entry points and pass environment variables through unchanged.

## Data and cache

The bundled data contains 10 scenes. The full dataset is described in `data/README.md`.

Build the after-network cache from `scenes/` and `bg/`:

```powershell
python code\after\prep_cache.py
```

Rebuild only the interface-sampling part of an existing cache:

```powershell
python code\after\prep_static.py
```

Optional variables:

| Variable | Meaning |
|---|---|
| `CACHE` | cache root, default `data/after/cache1` |
| `S_DIR` | scene directory, default `data/after/data1/scenes` |
| `B_DIR` | background directory, default `data/after/data1/bg` |
| `MEDIA` | scene range, e.g. `0,20` for `0..19` |
| `FORCE` | `1` overwrites existing files |

Each cache entry contains:

```text
D.npy      target log ratio
Yb.npy     background angular field
Pb.npy     background scalar flux
Pf.npy     reference scalar flux
meta.npz   IDX/OCC/P_BG/P_OBJ/GEO/TE
static.npz token positions and medium readings
```

The cache scripts only read precomputed scene files; they do not call the reference solver.

## Training

Front network:

```powershell
python code\front\train_front.py
```

After network:

```powershell
python code\after\train_after.py
```

Common environment variables:

| Module | Variable | Default |
|---|---|---|
| front | `NSTEP` | 100000 |
| front | `LR_FRONT` | 2e-3 |
| front | `FW`, `FL` | 384, 64 |
| front | `NSC_FRONT` | 2 |
| after | `F` | 192 |
| after | `NLAYER` | 6 |
| after | `NCROSS` | 2 |
| after | `NSTEP` | 100000 |
| after | `BATCH` | 16 |
| after | `NWORKERS` | 6 |

New training outputs are written to `weights/runs/`; `formal/` and `ablations/` are not overwritten.

## Evaluation protocol

- Front: `eval_front_arm.py` reports per-pulse relative L2 over the interior grid.
- After: `eval_after_arm.py` uses the reference background field and evaluates the correction.
- End-to-end: `eval_chain.py` evaluates front + after and reports per-pulse and per-scene RMSPE.
- Ablation: `eval_ablations.py` selects the checkpoint-specific output mode and switches, then writes `results/ablation_summary.json`.

The paper's formal per-scene result is `1.0445%` on the full test set.

## Ablation checkpoints

`weights/ablations/` contains the checkpoints for:

| Directory | Content |
|---|---|
| `tab5_6_loss/front/L_bg1`–`L_bg4` | front loss variants |
| `tab5_6_loss/after/L_pt1`–`L_pt5` | after loss variants |
| `tab5_7_prior/front/{with_prior,no_prior}` | analytic-prior ablation |
| `tab5_8_structure/single_stage` | single-stage network |
| `tab5_10_feat_front/front/*` | front input-feature ablations |
| `tab5_11_feat_after/after/*` | after input-feature ablations |
| `tab5_15_capacity/{3M,1M,300K,30K}` | capacity tiers |
