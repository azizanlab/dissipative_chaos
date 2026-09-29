<h1 align="center">Learning Dissipative Chaotic Dynamics with Boundedness Guarantees</h1>

<p align="center">
  Sunbochen Tang &nbsp;·&nbsp; Themistoklis Sapsis &nbsp;·&nbsp; Navid Azizan
  <br>
  <i>Massachusetts Institute of Technology</i>
  <br><br>
  <!-- <b>Proceedings of the National Academy of Sciences (PNAS), 2026</b> -->
  <br>
  Paper link coming soon
</p>

---

## Setup

```bash
conda create -n chaos_env python=3.9
conda activate chaos_env
pip install -r requirements.txt
pip install notebook
```

For NVIDIA RTX 50-series GPUs, replace the default PyTorch build with the
CUDA 12.8 build:

```bash
pip uninstall -y torch
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

## Reproducing the figures

| Notebook | System | Figures |
|---|---|---|
| [`eval/lorenz63.ipynb`](eval/lorenz63.ipynb) | Lorenz 63 | Fig. 3 |
| [`eval/lorenz96.ipynb`](eval/lorenz96.ipynb) | Lorenz 96 | Fig. 1C, Fig. 4A and B, SI Figs. S1 and S3 |
| [`eval/moore_spiegel.ipynb`](eval/moore_spiegel.ipynb) | Moore–Spiegel | Fig. 6, SI Figs. S5–S7 |

Each notebook loads the checkpoints in `assets/`, calls the functions in the
`eval/*.py` files and writes the figures to `figures/`. 

## Checkpoints and data

| Path under `assets/` | Content |
|---|---|
| `models/lorenz63/proj/` | `f*`(Proposed model) for Lorenz 63 |
| `models/lorenz96/proj/`, `noproj/` | `f*`(Proposed model) and `f̂`(Unconstrained model) for Lorenz 96 |
| `models/lorenz96/L96_test_20_20000.pkl` | 20 test rollout trajectories of 20,000 steps: ground truth, `f*` and `f̂` |
| `models/lorenz96/L96_energy_rollouts.npz` | energy time histories for SI Fig. S3 |
| `models/moore_spiegel/noproj/` | `f̂`(Unconstrained model) for Moore–Spiegel |
| `models/moore_spiegel/Vellip_proj/`, `Vquartic_proj/`, `Vmlp_proj/` | `f*`(Proposed model) with quadratic, quartic and MLP-PD `V` |
| `data/moore_spiegel/` | Moore–Spiegel ground-truth trajectories: 50 for training, 10 for testing |

## Training

Run the commands from the repository root. Models are saved under
`<system>/Experiment_results/`. The commands give the settings with which the
released checkpoints were trained.

```bash
# Lorenz 63
python src/train.py --system L63 --V_ellip --proj_flag --discrete \
    --num_epochs 30000 --train_M 20 --train_N 100 --train_sub_N 10 --stride 1 \
    --batch_size 128 --c_init 5.0 --reg_vol --lam_reg_vol 1e-8

# Lorenz 96
python src/train.py --system L96_5 --V_ellip --proj_flag --discrete --discrete_h 0.005 \
    --num_epochs 30000 --train_M 4 --train_N 500 --train_sub_N 2 --stride 1 \
    --c_init 100.0 --reg_vol --lam_reg_vol 1e-7

# Moore–Spiegel
DATA=assets/data/moore_spiegel/dim3_M50_N20000_dt0p01_amp10p0_T6p0_R20p0.npz
COMMON="--system Moore_Spiegel --load_data --data_loc $DATA --proj_flag --discrete \
    --num_epochs 30000 --train_M 10 --train_N 2000 --train_sub_N 2 --stride 1 \
    --c_init 100.0 --fix_c"

python src/train.py $COMMON --V_ellip   --reg_vol --lam_reg_vol 1e-6                 # quadratic V
python src/train.py $COMMON --V_quartic --reg_vol --lam_reg_vol 1e-7 --fix_alpha     # quartic V
python src/train.py $COMMON --V_MLP     --reg_vol --lam_reg_vol 1e-6 --fix_alpha \
    --lam_frac 0.01 --mc_tau 2.0 --lam_bump 0.1 --lam_anchor 0.001 --V_anchor_frac 0.25  # MLP-PD V
```

For the unconstrained model `f̂`, omit `--proj_flag`, `--reg_vol` and
`--lam_reg_vol`.