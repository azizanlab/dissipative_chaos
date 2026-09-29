#!/usr/bin/env python3
"""
Evaluation for the Lorenz 63 system. This script reproduces figures used in the paper regarding the Lorenz 63 experiments (Fig 3).
"""

import argparse
import os
import sys

# --- import path setup ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("src", "eval"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -----------------------------------------------------------------------------

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import stable_lorenz_model
from utils import gen_real_multi_traj, tensor_lorenz63

plt.rcParams.update({'font.size': 24})
fig_size = (10, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lorenz63_model(model_dir, checkpoint, proj_flag, discrete_dt, discrete_h):
    ckpt_path = os.path.join(model_dir, checkpoint)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    hf_dim = int(state_dict['fhat.0.weight'].shape[0]) if 'fhat.0.weight' in state_dict else 64
    params = {
        'state_dim': 3, 'odefun': tensor_lorenz63,
        'hf_dim': hf_dim, 'hV_dim': 64,
        'f_activation': 'GeLU', 'V_activation': 'ReLU', 'V_layer': 'None', 'V_scale': 10.0,
        'rehu_d': 0.01, 'eps_proj': 1e-4, 'proj_flag': proj_flag,
        'fix_c': False, 'c_init': 1.0, 'alpha_init': 1.0, 'fix_alpha': False, 'diagnose': False,
        'discrete': True, 'discrete_dt': discrete_dt, 'discrete_h': discrete_h,
        'V_MLP': False, 'V_ellip': True, 'V_quartic': False,
        'dt': discrete_dt,
    }
    model = stable_lorenz_model(params)
    res = model.load_state_dict(state_dict, strict=False)
    if res.missing_keys:
        print(f"[load] missing keys: {res.missing_keys}")
    if res.unexpected_keys:
        print(f"[load] unexpected keys: {res.unexpected_keys}")
    model.to(device).eval()
    return model, params


def generate_test_traj(model, Test_N, Test_M, dt, Test_xlim, params):
    """For Lorenz 63, since it's easy to simulate, the test data is generated on the fly."""
    print("Generating ground-truth trajectories")
    X_GT = gen_real_multi_traj(Test_M, Test_N, dt, odefun=params['odefun'],
                               x_lim=Test_xlim, x_dim=params['state_dim']).to(device)
    print("Rolling out the learned model")
    X_star = torch.zeros_like(X_GT).to(device)
    X_star[:, 0, :] = X_GT[:, 0, :]
    for i in tqdm(range(Test_N - 1)):
        if torch.max(torch.norm(X_star[:, i, :], p=2, dim=1)) > 1e8 or X_star[:, i, :].isnan().any():
            print(f"Learned trajectory grew unbounded / NaN at step {i}")
            break
        X_star[:, i + 1, :] = model(X_star[:, i, :])
    return X_GT, X_star


def _ellipsoid_samples(model, state_dim, n_points=10000):
    """Sample points on the invariant ellipsoid {(x-x0)^T Q (x-x0) = c^2} for visualization."""
    x0 = model.V.x_0.detach().cpu().numpy().squeeze()
    Q = model.V._construct_Q().detach().cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(Q)
    if np.min(eigvals) <= 1e-3:
        print(f"Q nearly singular (min eig {np.min(eigvals):.2e}); skipping invariant-set overlay.")
        return None
    c2 = (model.c.detach().cpu().numpy()) ** 2
    u = np.random.randn(state_dim, n_points)
    u /= np.linalg.norm(u, axis=0)
    pts = eigvecs @ (np.sqrt(c2 / eigvals)[:, None] * u)
    return pts[0] + x0[0], pts[1] + x0[1], pts[2] + x0[2], x0


def plot_attractor_projections(X_GT, X_star, model, params, out_dir):
    Test_M = X_GT.shape[0]
    ell = _ellipsoid_samples(model, params['state_dim']) if params['proj_flag'] else None

    fig, ax = plt.subplots(Test_M, 3, figsize=(fig_size[0] * 3, fig_size[1] * Test_M))
    ax = np.atleast_2d(ax)
    for m in range(Test_M):
        for i in range(3):
            if i == 2:
                ax[m, i].plot(X_GT[m, :, i].detach().cpu().numpy(), X_GT[m, :, 0].detach().cpu().numpy(), label='GT')
                ax[m, i].plot(X_star[m, :, i].detach().cpu().numpy(), X_star[m, :, 0].detach().cpu().numpy(), label='fstar')
                ax[m, i].set_xlabel(r'$x_3$'); ax[m, i].set_ylabel(r'$x_1$')
            else:
                ax[m, i].plot(X_GT[m, :, i].detach().cpu().numpy(), X_GT[m, :, i + 1].detach().cpu().numpy(), label='GT')
                ax[m, i].plot(X_star[m, :, i].detach().cpu().numpy(), X_star[m, :, i + 1].detach().cpu().numpy(), label='fstar')
                ax[m, i].set_xlabel(fr'$x_{i+1}$'); ax[m, i].set_ylabel(fr'$x_{i+2}$')
            ax[m, i].legend()
            if ell is not None:
                x, y, z, x0 = ell
                if i == 0:
                    ax[m, i].scatter(x, y, color='y', alpha=0.3, s=1); ax[m, i].scatter(x0[0], x0[1], color='r', s=50)
                elif i == 1:
                    ax[m, i].scatter(y, z, color='y', alpha=0.3, s=1); ax[m, i].scatter(x0[1], x0[2], color='r', s=50)
                else:
                    ax[m, i].scatter(z, x, color='y', alpha=0.3, s=1); ax[m, i].scatter(x0[2], x0[0], color='r', s=50)
    fig.tight_layout()
    out = os.path.join(out_dir, 'Ellip_views.png')
    plt.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved Fig 3A to {out}")


def plot_V_history(X_GT, X_star, model, params, out_dir):
    dt = params['dt']
    Test_M, Test_N = X_GT.shape[0], X_GT.shape[1]
    V_GT = torch.zeros(Test_M, Test_N, device=device)
    V_star = torch.zeros(Test_M, Test_N, device=device)
    with torch.no_grad():
        for i in tqdm(range(Test_N)):
            V_GT[:, i] = model.V(X_GT[:, i, :]).squeeze(1)
            V_star[:, i] = model.V(X_star[:, i, :]).squeeze(1)
    fig, ax = plt.subplots(Test_M, 1, figsize=(fig_size[0] * 2, fig_size[1] * Test_M))
    ax = np.atleast_1d(ax)
    t = np.arange(0, Test_N * dt, dt)[:Test_N]
    c2 = float(model.c.detach().cpu()) ** 2
    for m in range(Test_M):
        ax[m].plot(t, V_GT[m, :].detach().cpu().numpy(), label='GT')
        ax[m].plot(t, V_star[m, :].detach().cpu().numpy(), label='fstar')
        ax[m].hlines(c2, 0, t[-1], 'k', linewidth=2, label=r'$c^2$')
        ax[m].legend(); ax[m].set_xlabel('t'); ax[m].set_ylabel('V'); ax[m].set_yscale('log')
    fig.tight_layout()
    out = os.path.join(out_dir, 'V_test.png')
    plt.savefig(out); plt.close(fig)
    print(f"Saved Fig 3C to {out}")


def plot_flow_map(model, params, out_dir, grid_size=20):
    x0 = model.V.x_0.detach().cpu().numpy().squeeze()
    Q = model.V._construct_Q().detach().cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.clip(eigvals, 1e-12, None)
    c2 = (model.c.detach().cpu().numpy()) ** 2
    n_points = 10000
    u = np.random.randn(params['state_dim'], n_points); u /= np.linalg.norm(u, axis=0)
    pts = eigvecs @ (np.sqrt(c2 / eigvals)[:, None] * u)
    x, y, z = pts[0] + x0[0], pts[1] + x0[1], pts[2] + x0[2]

    def _rng(a):
        mid = (a.max() + a.min()) / 2
        half = (a.max() - a.min())
        return mid - half, mid + half
    xx = np.linspace(*_rng(x), grid_size)
    yy = np.linspace(*_rng(y), grid_size)
    zz = np.linspace(*_rng(z), grid_size)
    X, Y, Z = np.meshgrid(xx, yy, zz)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten(), Z.flatten()]).T, dtype=torch.float32, device=device)

    with torch.no_grad():
        gt = tensor_lorenz63(grid).detach().cpu().numpy()
    U1, V1 = gt[:, 0].reshape(X.shape), gt[:, 1].reshape(Y.shape)
    if params['proj_flag']:
        mo = model.f_proj(grid).detach().cpu().numpy()
    else:
        mo = model.fhat(grid).detach().cpu().numpy()
    U2, V2 = mo[:, 0].reshape(X.shape), mo[:, 1].reshape(Y.shape)

    k = grid_size // 2
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].quiver(X[:, :, k], Y[:, :, k], U1[:, :, k], V1[:, :, k], color='b', alpha=0.5)
    axs[0].set_title('Lorenz 63'); axs[0].set_xlabel('X'); axs[0].set_ylabel('Y'); axs[0].set_aspect('equal')
    axs[1].quiver(X[:, :, k], Y[:, :, k], U2[:, :, k], V2[:, :, k], color='r', alpha=0.5)
    axs[1].scatter(x, y, color='y', alpha=0.3, s=1); axs[1].scatter(x0[0], x0[1], color='r', s=50)
    axs[1].set_title('Trained Model'); axs[1].set_xlabel('X'); axs[1].set_ylabel('Y'); axs[1].set_aspect('equal')
    fig.tight_layout()
    out = os.path.join(out_dir, 'flow_map_projection.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved Fig 3B to {out}")


def main():
    p = argparse.ArgumentParser(description="Reproduce the Lorenz-63 paper figure (Fig 3).")
    p.add_argument('--model_dir', type=str, default='assets/models/lorenz63/proj')
    p.add_argument('--checkpoint', type=str, default='E29999.pt')
    p.add_argument('--out', type=str, default='figures/lorenz63')
    p.add_argument('--proj_flag', action='store_true', default=True,
                   help='Model was trained with projection (default True for the paper model).')
    p.add_argument('--no_proj', dest='proj_flag', action='store_false')
    p.add_argument('--Test_N', type=int, default=50000)
    p.add_argument('--Test_M', type=int, default=1)
    p.add_argument('--Test_xlim', type=float, default=50.0)
    p.add_argument('--dt', type=float, default=0.01, help='integration / sampling step')
    p.add_argument('--discrete_h', type=float, default=0.01, help='RK4 sub-step inside the model')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    model, params = load_lorenz63_model(args.model_dir, args.checkpoint,
                                        args.proj_flag, args.dt, args.discrete_h)
    X_GT, X_star = generate_test_traj(model, args.Test_N, args.Test_M, args.dt,
                                      args.Test_xlim, params)
    plot_attractor_projections(X_GT, X_star, model, params, args.out)   # Fig 3A
    plot_flow_map(model, params, args.out)  # Fig 3B
    plot_V_history(X_GT, X_star, model, params, args.out)   # Fig 3C
    print(f"Lorenz-63 figure panels written to {args.out}/")


if __name__ == '__main__':
    main()
