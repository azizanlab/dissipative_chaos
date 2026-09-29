#!/usr/bin/env python3
"""
This script computes and caches the testing rollouts and metrics for the Moore-Spiegel comparison experiment. 

The figure generation code is V_ablation_MS.py, which then loads the cached rollouts directly.
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
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

from model import stable_lorenz_model
from utils import tensor_moore_spiegel, rk4_model


# model loading helpers
def detect_V_type(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith('V.log_diag_LA') for k in keys):
        return 'quartic'
    if any(k.startswith('V.f.') for k in keys):
        return 'mlp'
    return 'ellip'


def detect_dims(state_dict):
    hf_dim = int(state_dict['fhat.0.weight'].shape[0])
    hV_dim = None
    v_layer = 'None'
    v_scale = 10.0
    if 'V.f.0.weight' in state_dict:
        hV_dim = int(state_dict['V.f.0.weight'].shape[0])
        if 'V.f.6.weight' in state_dict:
            v_layer = 'Tanh'
            v_scale = float(state_dict['V.f.6.weight'].item())
    return hf_dim, hV_dim, v_layer, v_scale


def load_model(model_dir, checkpoint, device, proj_flag=False):
    ckpt_path = os.path.join(model_dir, checkpoint)
    state_dict = torch.load(ckpt_path, map_location=device)
    v_type = detect_V_type(state_dict)
    hf_dim, hV_dim, v_layer, v_scale = detect_dims(state_dict)
    params = {
        'state_dim': 3,
        'proj_flag': proj_flag,
        'discrete': True,
        'discrete_dt': 0.01,
        'discrete_h': 0.01,
        'hf_dim': hf_dim,
        'hV_dim': hV_dim if hV_dim is not None else 64,
        'f_activation': 'GeLU',
        'V_activation': 'ReLU',
        'V_layer': v_layer,
        'V_scale': v_scale,
        'rehu_d': 0.01,
        'V_MLP':     v_type == 'mlp',
        'V_ellip':   v_type == 'ellip',
        'V_quartic': v_type == 'quartic',
        'eps_proj': 1e-4,
        'fix_c': False,
        'c_init': 100.0,
        'alpha_init': 1.0,
        'fix_alpha': False,
        'diagnose': False,
    }
    model = stable_lorenz_model(params)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model, v_type


# rollout compute helpers

def rollout(model, x0, steps, device):
    x = torch.tensor(x0, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    traj = [x.cpu().numpy()]
    with torch.no_grad():
        for _ in tqdm(range(steps), desc='rollout'):
            x = model(x)
            traj.append(x.cpu().numpy())
    traj = np.stack(traj, axis=0)
    return traj.transpose(1, 0, 2)


def sample_sublevel_set_affine(model, device, c2, n_samples=200000,
                               seed=0, chunk=4096):
    model.eval()
    with torch.no_grad():
        Q = model.V._construct_Q().detach().cpu().numpy()
        x_0 = model.V.x_0.squeeze(-1).detach().cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(Q)
    L = (eigvecs * (1.0 / np.sqrt(eigvals))[None, :]).astype(np.float32)

    rng = np.random.default_rng(seed)
    N = rng.standard_normal(size=(n_samples, 3)).astype(np.float32)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    r = rng.random(size=(n_samples, 1)).astype(np.float32) ** (1.0 / 3.0)
    u = N * r
    y = (u @ L.T) * np.float32(np.sqrt(c2))
    x = y + x_0[None, :].astype(np.float32)

    v_all = np.empty(n_samples, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_samples, chunk):
            t = torch.from_numpy(x[s:s + chunk]).to(device)
            v_all[s:s + chunk] = model.V(t).squeeze(-1).cpu().numpy()
    keep = v_all <= c2
    return x[keep], {
        'n_samples': n_samples,
        'n_in_sublevel': int(keep.sum()),
        'frac_in_sublevel': float(keep.mean()),
    }


def pooled_state_pairs(batch, transient_steps, i, j):
    return batch[:, transient_steps:, [i, j]].reshape(-1, 2)


def hist2d_kl_js(pts_gt, pts_pred, bins, lim, eps=1e-9):
    H_gt, xe, ye = np.histogram2d(pts_gt[:, 0], pts_gt[:, 1],
                                  bins=bins, range=lim, density=True)
    H_pr, _, _ = np.histogram2d(pts_pred[:, 0], pts_pred[:, 1],
                                bins=bins, range=lim, density=True)
    bin_area = ((xe[-1] - xe[0]) / bins) * ((ye[-1] - ye[0]) / bins)
    p = (H_gt * bin_area).ravel()
    q = (H_pr * bin_area).ravel()
    p = p / p.sum()
    q = q / q.sum()
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    KL = float(np.sum(p * np.log(p / q)))
    JS = float(jensenshannon(p, q))
    return H_gt, H_pr, xe, ye, KL, JS


def load_cached_trajs(results_path):
    if not os.path.exists(results_path):
        return None
    print(f"Found cached results at {results_path}; loading instead of rolling out.")
    npz = np.load(results_path, allow_pickle=True)
    out = {'true': npz['true'], 'pred': npz['pred']}
    for key in ('dt', 'T_par', 'R_par'):
        if key in npz.files:
            out[key] = float(npz[key])
    npz.close()
    return out


# trajectory computation for mirrored pairs of initial conditions
def compute_mirror_trajectories(model, dt, T_par, R_par, device,
                                n_pairs=3, T_total=200.0, transient=20.0,
                                ic_lim=1.0, seed=7):
    """Roll out the model and the true Moore-Spiegel ODE from n_pairs random
    initial conditions and their mirrors (x0, -x0). Returns (X0, true, pred)
    numpy arrays, each trajectory of shape (2*n_pairs, n_steps+1, 3). These
    feed the density row of the comparison figure; plotting lives in V_ablation_MS."""
    n_steps = int(T_total / dt)
    g = torch.Generator(device=device).manual_seed(seed)
    X0_pos = (torch.rand(n_pairs, 3, generator=g, device=device) * 2 * ic_lim) - ic_lim
    X0 = torch.empty(2 * n_pairs, 3, device=device)
    X0[0::2] = X0_pos
    X0[1::2] = -X0_pos

    true = torch.zeros(2 * n_pairs, n_steps + 1, 3, device=device)
    true[:, 0] = X0
    for i in range(n_steps):
        true[:, i + 1] = rk4_model(tensor_moore_spiegel, dt,
                                   y=true[:, i], T=T_par, R=R_par)

    pred = torch.zeros(2 * n_pairs, n_steps + 1, 3, device=device)
    pred[:, 0] = X0
    with torch.no_grad():
        for i in range(n_steps):
            pred[:, i + 1] = model(pred[:, i])
            if torch.isnan(pred[:, i + 1]).any():
                print(f"  mirror: model NaN at step {i + 1}")
                break
    return X0.cpu().numpy(), true.cpu().numpy(), pred.cpu().numpy()


def compute_and_cache(model_dir, checkpoint, data_loc, proj_flag, device,
                      out_dir=None, rollout_steps=None,
                      n_pairs=3, mirror_T_total=200.0, mirror_transient=20.0,
                      mirror_ic_lim=1.0, mirror_seed=7, force=False):
    if out_dir is None:
        out_dir = os.path.dirname(model_dir.rstrip('/'))
    ckpt_stem = checkpoint.replace('.pt', '')
    results_path = os.path.join(out_dir, f'eval_{ckpt_stem}_results.npz')
    mirror_path = os.path.join(out_dir, f'eval_{ckpt_stem}_mirror.npz')
    if (not force) and os.path.isfile(results_path) and os.path.isfile(mirror_path):
        print(f"[eval_MS] caches present, skipping: {out_dir}")
        return results_path, mirror_path

    data = np.load(data_loc)
    test_data = data['test_data']
    dt = float(data['dt'])
    T_par = float(data['T'])
    R_par = float(data['R'])

    model, v_type = load_model(model_dir, checkpoint, device, proj_flag=proj_flag)
    print(f"[eval_MS] {out_dir}: V={v_type}, proj={proj_flag}")

    # Test-set rollout cache.
    if force or not os.path.isfile(results_path):
        steps = rollout_steps or (test_data.shape[1] - 1)
        pred = rollout(model, test_data[:, 0, :], steps, device)
        true = test_data[:, :steps + 1, :]
        np.savez(results_path, true=true, pred=pred, dt=dt,
                 T_par=T_par, R_par=R_par, checkpoint=checkpoint)
        print(f"  saved {results_path}")

    # Mirror-pair cache.
    if force or not os.path.isfile(mirror_path):
        X0, m_true, m_pred = compute_mirror_trajectories(
            model, dt, T_par, R_par, device, n_pairs=n_pairs,
            T_total=mirror_T_total, transient=mirror_transient,
            ic_lim=mirror_ic_lim, seed=mirror_seed)
        np.savez_compressed(
            mirror_path, X0=X0, true=m_true, pred=m_pred, dt=dt,
            T_par=T_par, R_par=R_par, T_total=mirror_T_total,
            transient=mirror_transient)
        print(f"  saved {mirror_path}")

    return results_path, mirror_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Moore-Spiegel compute + cache (no plotting).")
    p.add_argument('--model_dir', type=str, required=True,
                   help="Directory containing the checkpoint (the .../models dir).")
    p.add_argument('--data_loc', type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='E29999.pt')
    p.add_argument('--proj_flag', action='store_true',
                   help='Set if the model was trained with projection.')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Where to write caches (default: parent of --model_dir).')
    p.add_argument('--rollout_steps', type=int, default=None)
    p.add_argument('--n_pairs', type=int, default=3)
    p.add_argument('--mirror_T_total', type=float, default=200.0)
    p.add_argument('--mirror_transient', type=float, default=20.0)
    p.add_argument('--mirror_ic_lim', type=float, default=1.0)
    p.add_argument('--mirror_seed', type=int, default=7)
    p.add_argument('--force', action='store_true')
    args = p.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_and_cache(
        args.model_dir, args.checkpoint, args.data_loc, args.proj_flag, device,
        out_dir=args.out_dir, rollout_steps=args.rollout_steps,
        n_pairs=args.n_pairs, mirror_T_total=args.mirror_T_total,
        mirror_transient=args.mirror_transient, mirror_ic_lim=args.mirror_ic_lim,
        mirror_seed=args.mirror_seed, force=args.force)


if __name__ == '__main__':
    main()
