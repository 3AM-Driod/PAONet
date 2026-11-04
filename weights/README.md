# Weights

This directory contains the formal PAONet checkpoints and the ablation checkpoints used in
the paper. All files are saved as PyTorch `state_dict` objects.

## Directory layout

```text
weights/
├─ formal/
│  ├─ front/front_net.pt
│  └─ after/after_net.pt
└─ ablations/
   ├─ tab5_6_loss/
   │  ├─ front/L_bg1 ... L_bg4
   │  └─ after/L_pt1 ... L_pt5
   ├─ tab5_7_prior/front/{with_prior,no_prior}
   ├─ tab5_8_structure/single_stage
   ├─ tab5_10_feat_front/front/*
   ├─ tab5_11_feat_after/after/*
   └─ tab5_15_capacity/{3M,1M,300K,30K}
```

## Formal checkpoints

| Path | Parameters | Description |
|---|---:|---|
| `formal/front/front_net.pt` | 419,976 | background-response network |
| `formal/after/after_net.pt` | 2,494,018 | perturbation-response network |
| **Total** | **2,913,994** | Table 5.16: `Ours / 2,913,994 / 1.0445` |

The per-scene mean on the full test set is **1.0445%**.

End-to-end evaluation:

```powershell
python code\eval_chain.py
```

## Ablation checkpoints

| Directory | Content |
|---|---|
| `ablations/tab5_6_loss/front/L_bg1`–`L_bg4` | front loss variants |
| `ablations/tab5_6_loss/after/L_pt1`–`L_pt5` | after loss variants |
| `ablations/tab5_7_prior/front/{with_prior,no_prior}` | analytic-prior ablation |
| `ablations/tab5_8_structure/single_stage` | single-stage network |
| `ablations/tab5_10_feat_front/front/*` | front input-feature ablations |
| `ablations/tab5_11_feat_after/after/*` | after input-feature ablations |
| `ablations/tab5_15_capacity/{3M,1M,300K,30K}` | capacity tiers |

Evaluate all ablation checkpoints:

```powershell
python code\eval_ablations.py --kind after
python code\eval_ablations.py --kind front
python code\eval_ablations.py --kind all
```

`eval_ablations.py` selects the checkpoint-specific output mode and feature switches.
Results are written to `results/ablation_summary.json`; per-case logs are in
`results/ablation_logs/`.

The comparison-model checkpoints for Table 5.16 (`tab5_16_compare`) are not part of the
ablation set and are not included here.

## Model configuration

| Module | Configuration |
|---|---|
| Front | `FW=384`, `FL=64`, `NSC_FRONT=2` |
| After | `F=192`, `NLAYER=6`, `NCROSS=2`, `NHEAD=4` |

Training and inference build the networks with these defaults unless the corresponding
environment variables are set explicitly.
