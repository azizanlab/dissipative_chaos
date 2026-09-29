#!/usr/bin/env python3
"""
This script reads from cached rollout trajectories to produce Fig. 6 in the paper. The figure presents the ablation study comparing different Moore-Spiegel models with quadratic, quartic, and MLP-PD V parameterizations.
"""

import argparse
import glob
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# --- import path setup ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("src", "eval"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -----------------------------------------------------------------------------


from eval_MS import (
    load_model,
    rollout,
    load_cached_trajs,
    sample_sublevel_set_affine,
    hist2d_kl_js,
    pooled_state_pairs,
    compute_and_cache,
)


# Specific text parameters
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 30,
    "font.weight": "normal",
    "axes.labelsize": 26,
    "axes.labelweight": "normal",
    "axes.titlesize": 34,
    "axes.titleweight": "normal",
    "legend.fontsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
})


VARIANTS = [
    ('Unconstrained',
     'noproj',
     False),
    (r'Quadratic $V$',
     'Vellip_proj',
     True),
    (r'Quartic $V$',
     'Vquartic_proj',
     True),
    (r'MLP-PD $V$',
     'Vmlp_proj',
     True),
]

# Level-set opacity parameter for different configurations
ALPHA_BY_TITLE = {
    'Unconstrained':    0.25,   # no ellipsoid anyways
    r'Quadratic $V$':   0.10,
    r'Quartic $V$':     0.10,
    r'MLP-PD $V$':      0.25,
}


def _alpha_for(title, fallback):
    return ALPHA_BY_TITLE.get(title, fallback)


def find_model_dir(root, pattern):
    matches = sorted(glob.glob(os.path.join(root, pattern)))
    if len(matches) == 0:
        raise FileNotFoundError(f"No match for pattern: {pattern!r} under {root!r}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Pattern {pattern!r} matched {len(matches)} dirs (expected one):\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def filter_finite(arr):
    return arr[np.all(np.isfinite(arr), axis=1)]


def panels_to_npz(panels, cache_path, meta):
    """Cache each variant's shell points to avoid resampling."""
    flat = {f'meta_{k}': np.array(v) for k, v in meta.items()}
    flat['titles'] = np.array([pn['title'] for pn in panels])
    for i, pn in enumerate(panels):
        flat[f'v_type_{i}']   = np.array(pn['v_type'])
        flat[f'proj_flag_{i}'] = np.array(pn['proj_flag'])
        flat[f'c2_{i}']        = np.array(pn['c2'])
        flat[f'vol_frac_{i}']  = np.array(pn['vol_frac'])
        flat[f'shell_pts_{i}'] = pn['shell_pts'].astype(np.float32)
        flat[f'pts_t_xy_{i}']  = pn['pts_t_xy'].astype(np.float32)
        flat[f'pts_p_xy_{i}']  = pn['pts_p_xy'].astype(np.float32)
        flat[f'pts_t_xz_{i}']  = pn['pts_t_xz'].astype(np.float32)
        flat[f'pts_p_xz_{i}']  = pn['pts_p_xz'].astype(np.float32)
    np.savez_compressed(cache_path, **flat)


def panels_from_npz(cache_path, meta):
    if not os.path.isfile(cache_path):
        return None
    npz = np.load(cache_path, allow_pickle=False)
    try:
        for k, v in meta.items():
            cached = npz[f'meta_{k}'].item()
            if isinstance(v, float):
                if not np.isclose(cached, v):
                    print(f"[cache] miss: meta {k}={cached} != {v}")
                    return None
            elif cached != v:
                print(f"[cache] miss: meta {k}={cached!r} != {v!r}")
                return None
        titles = npz['titles']
        panels = []
        for i in range(len(titles)):
            panels.append({
                'title':     str(titles[i]),
                'v_type':    str(npz[f'v_type_{i}']),
                'proj_flag': bool(npz[f'proj_flag_{i}']),
                'c2':        float(npz[f'c2_{i}']),
                'vol_frac':  float(npz[f'vol_frac_{i}']),
                'shell_pts': npz[f'shell_pts_{i}'],
                'pts_t_xy':  npz[f'pts_t_xy_{i}'],
                'pts_p_xy':  npz[f'pts_p_xy_{i}'],
                'pts_t_xz':  npz[f'pts_t_xz_{i}'],
                'pts_p_xz':  npz[f'pts_p_xz_{i}'],
            })
        return panels
    finally:
        npz.close()


def sample_boundary_shell(model, device, n_samples, shell_frac, seed):
    c2 = (model.c ** 2).detach().cpu().item()
    pts_in, stats = sample_sublevel_set_affine(
        model, device, c2, n_samples=n_samples, seed=seed)
    if shell_frac <= 0.0:
        return pts_in, stats['frac_in_sublevel'], stats['n_in_sublevel'], c2
    with torch.no_grad():
        v_kept = []
        chunk = 4096
        for s in range(0, len(pts_in), chunk):
            t = torch.from_numpy(pts_in[s:s + chunk].astype(np.float32)).to(device)
            v_kept.append(model.V(t).squeeze(-1).cpu().numpy())
    v_kept = np.concatenate(v_kept) if v_kept else np.empty(0, dtype=np.float32)
    shell_pts = pts_in[v_kept >= shell_frac * c2]
    return shell_pts, stats['frac_in_sublevel'], stats['n_in_sublevel'], c2


def _range_of(point_clouds, dim, pad=0.05):
    arrays = [pc[:, dim] for pc in point_clouds if len(pc)]
    if not arrays:
        return [-1.0, 1.0]
    v_min = min(a.min() for a in arrays)
    v_max = max(a.max() for a in arrays)
    p = pad * (v_max - v_min if v_max > v_min else 1.0)
    return [float(v_min - p), float(v_max + p)]


def _build_density_data(args, traj_map):
    assert len(traj_map) == len(VARIANTS)
    data = np.load(args.data)
    dt_test = float(data['dt'])
    n_steps_test = data['test_data'].shape[1] - 1
    transient_test = min(int(args.transient_time / dt_test), n_steps_test - 1)

    gt_pooled = None
    per_variant = []
    for (title, pattern, proj_flag), traj_idx in zip(VARIANTS, traj_map):
        model_root = find_model_dir(args.root, pattern)
        ckpt_stem = args.checkpoint.replace('.pt', '')
        if proj_flag:
            mirror_path = os.path.join(
                model_root, f'eval_{ckpt_stem}_mirror.npz')
            if not os.path.isfile(mirror_path):
                raise FileNotFoundError(
                    f"Mirror cache missing for {title}: {mirror_path}\n"
                    f"  Run eval_MS.py on this model to generate it.")
            npz = np.load(mirror_path)
            true_arr = npz['true']
            pred_arr = npz['pred']
            dt_m = float(npz['dt']); transient_m = float(npz['transient'])
            npz.close()
            skip_m = max(0, min(int(transient_m / dt_m), true_arr.shape[1] - 2))
            pred_pts = pred_arr[:, skip_m:, :].reshape(-1, 3)
            true_pts = true_arr[:, skip_m:, :].reshape(-1, 3)
            pred_full = pred_arr.reshape(-1, 3)
            if gt_pooled is None:
                gt_pooled = true_pts
            source = 'mirror_pair_test'
            label = title
        else:
            results_path = os.path.join(
                model_root, f'eval_{ckpt_stem}_results.npz')
            if not os.path.isfile(results_path):
                raise FileNotFoundError(
                    f"Per-model trajectory cache missing for {title}: "
                    f"{results_path}")
            npz = np.load(results_path)
            true_arr = npz['true'][traj_idx].astype(np.float32)
            pred_arr = npz['pred'][traj_idx].astype(np.float32)
            npz.close()
            pred_pts = pred_arr[transient_test:, :]
            true_pts = true_arr[transient_test:, :]
            
            pred_full = pred_arr
            source = f'test_data[traj {traj_idx}]'
            label = title
        per_variant.append({
            'title': title, 'proj_flag': proj_flag,
            'pred_pts': pred_pts, 'true_pts': true_pts,
            'pred_full': pred_full,
            'source': source, 'panel_label': label,
        })

    if gt_pooled is None:
        gt_pooled = per_variant[0]['true_pts']
    return per_variant, gt_pooled


def _plot_combined_levelset_density(panels, full_trajs, traj_map,
                                    per_variant_density, gt_pooled,
                                    fig_xlim, fig_ylim, args, out_path, rng,
                                    density_dim_pair=(0, 1), bins=80,
                                    color_scale='log'):
    assert len(traj_map) == len(panels)
    n_variants = len(panels)
    n_main = 1 + n_variants

    fig = plt.figure(figsize=(10 * n_main + 1.2, 20.4))
    gs = fig.add_gridspec(2, n_main + 1,
                          width_ratios=[1.0] * n_main + [0.04],
                          wspace=0.20, hspace=0.18)

    ax_gt_top = fig.add_subplot(gs[0, 0])
    seen_idx = []
    for ft, traj_idx in zip(full_trajs, traj_map):
        if traj_idx in seen_idx:
            continue
        seen_idx.append(traj_idx)
        skip = ft['transient']
        t_traj = ft['true'][traj_idx]
        _plot_traj_line(ax_gt_top, t_traj, f'GT (traj {traj_idx})',
                        skip, zorder=2, color='C0')
    ax_gt_top.set_xlim(fig_xlim); ax_gt_top.set_ylim(fig_ylim)
    ax_gt_top.set_xlabel(DIM_LABELS[0]); ax_gt_top.set_ylabel(DIM_LABELS[1])
    ax_gt_top.set_title('Ground Truth')

    for k, (pn, ft, traj_idx) in enumerate(zip(panels, full_trajs, traj_map)):
        if pn['proj_flag']:
            ax = fig.add_subplot(gs[0, k + 1], sharey=ax_gt_top)
        else:
            ax = fig.add_subplot(gs[0, k + 1])
        skip = ft['transient']
        t_traj = ft['true'][traj_idx]; p_traj = ft['pred'][traj_idx]
        _draw_shell_2d(ax, pn['shell_pts'], 0, 1, args, rng,
                       alpha=_alpha_for(pn['title'], args.shell_alpha))
        _plot_traj_line(ax, t_traj, 'Ground Truth', skip, zorder=2, color='C0')
        _plot_traj_line(ax, p_traj, 'Prediction',   skip, zorder=3, color='C1')
        if pn['proj_flag']:
            ax.set_xlim(fig_xlim); ax.set_ylim(fig_ylim)
        else:
            tp_xy = np.concatenate([t_traj[skip:, :2], p_traj[skip:, :2]])
            tp_xy = tp_xy[np.all(np.isfinite(tp_xy), axis=1)]
            ax.set_xlim(_range_of([tp_xy], 0))
            ax.set_ylim(_range_of([tp_xy], 1))
        ax.set_xlabel(DIM_LABELS[0])
        ax.set_title(pn['title'])

    i, j = density_dim_pair
    xlab, ylab = DIM_LABELS[i], DIM_LABELS[j]


    pts_p_per = []
    for vd in per_variant_density:
        src = vd['pred_full'] if not vd['proj_flag'] else vd['pred_pts']
        pts_p_per.append(src[:, [i, j]])

    gt_2d = gt_pooled[:, [i, j]]
    gt_2d = gt_2d[np.isfinite(gt_2d).all(axis=1)]
    if not len(gt_2d):
        raise RuntimeError(f"No finite GT points for dims={density_dim_pair}")
    x_lo, x_hi = float(gt_2d[:, 0].min()), float(gt_2d[:, 0].max())
    y_lo, y_hi = float(gt_2d[:, 1].min()), float(gt_2d[:, 1].max())
    pad_x = 0.05 * (x_hi - x_lo if x_hi > x_lo else 1.0)
    pad_y = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    dlim = [[x_lo - pad_x, x_hi + pad_x], [y_lo - pad_y, y_hi + pad_y]]
    if args.density_xlim:
        try:
            override = [float(s) for s in args.density_xlim.split(',')]
            if len(override) == 2:
                dlim[0] = override
        except ValueError:
            pass
    if args.density_ylim and density_dim_pair == (0, 1):
        try:
            override = [float(s) for s in args.density_ylim.split(',')]
            if len(override) == 2:
                dlim[1] = override
        except ValueError:
            pass
    bin_area = (((dlim[0][1] - dlim[0][0]) / bins)
                * ((dlim[1][1] - dlim[1][0]) / bins))

    def _dh(points_2d):
        if len(points_2d) == 0:
            return np.zeros((bins, bins))
        n_full = len(points_2d)
        finite_mask = np.isfinite(points_2d).all(axis=1)
        points = points_2d[finite_mask]
        if len(points) == 0:
            return np.zeros((bins, bins))
        counts, _, _ = np.histogram2d(points[:, 0], points[:, 1],
                                      bins=bins, range=dlim)
        return counts / (n_full * bin_area)

    H_gt = _dh(gt_2d)
    H_preds = [_dh(pp) for pp in pts_p_per]
    H_for_vmax = [H_gt]
    for vd, H_p in zip(per_variant_density, H_preds):
        if vd['proj_flag']:
            H_for_vmax.append(H_p)
    vmax = max(max(float(H.max()) for H in H_for_vmax), 1e-6)
    norm = _make_density_norm(color_scale, vmax)
    extent = [dlim[0][0], dlim[0][1], dlim[1][0], dlim[1][1]]
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(cmap(0.0))

    ax_gt_bot = fig.add_subplot(gs[1, 0])
    im = ax_gt_bot.imshow(H_gt.T, origin='lower', extent=extent, norm=norm,
                          aspect='auto', cmap=cmap)
    ax_gt_bot.set_xlabel(xlab); ax_gt_bot.set_ylabel(ylab)

    for k, (vd, H_p) in enumerate(zip(per_variant_density, H_preds)):
        ax = fig.add_subplot(gs[1, k + 1], sharey=ax_gt_bot)
        im = ax.imshow(H_p.T, origin='lower', extent=extent, norm=norm,
                       aspect='auto', cmap=cmap)
        ax.set_xlabel(xlab)

    cax = fig.add_subplot(gs[1, n_main])
    fig.colorbar(im, cax=cax)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               color='red', alpha=0.6, markersize=24,
               markeredgecolor='none', label='Sublevel set'),
        Line2D([0], [0], color='C0', lw=8, label='Ground Truth'),
        Line2D([0], [0], color='C1', lw=8, label='Prediction'),
    ]
    fig.legend(handles=handles,
               loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=True, fontsize=42,
               handletextpad=0.8, columnspacing=3.5)

    fig.canvas.draw()
    box_top = ax_gt_top.get_position()
    box_bot = ax_gt_bot.get_position()
    label_x = max(box_top.x0 - 0.025, 0.002)
    fig.text(label_x, box_top.y1 + 0.005, 'A',
             fontsize=90, va='bottom', ha='left')
    fig.text(label_x, box_bot.y1 + 0.005, 'B',
             fontsize=90, va='bottom', ha='left')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined figure: {out_path}")


def _make_density_norm(scale, vmax):
    if scale == 'log':
        floor = max(vmax * 1e-4, 1e-12)
        return mcolors.LogNorm(vmin=floor, vmax=vmax)
    if scale == 'sqrt':
        return mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    return mcolors.Normalize(vmin=0.0, vmax=vmax)


def _load_full_trajs(args):
    out = []
    for title, pattern, proj_flag in VARIANTS:
        model_root = find_model_dir(args.root, pattern)
        results_path = os.path.join(
            model_root, f'eval_{args.checkpoint.replace(".pt","")}_results.npz')
        if not os.path.isfile(results_path):
            raise FileNotFoundError(
                f"Per-model trajectory cache missing for single-traj mode: "
                f"{results_path}")
        npz = np.load(results_path)
        true = npz['true']; pred = npz['pred']
        dt = float(npz['dt'])
        npz.close()
        transient_steps = min(int(args.transient_time / dt), true.shape[1] - 2)
        out.append({
            'title': title, 'proj_flag': proj_flag,
            'true': true, 'pred': pred, 'transient': transient_steps,
        })
    return out


DIM_LABELS = (r'$x$', r'$\dot x$', r'$\ddot x$')


def _maybe_subsample(arr, cap, rng):
    if len(arr) <= cap:
        return arr
    idx = rng.choice(len(arr), cap, replace=False)
    return arr[idx]


def _draw_shell_2d(ax, shell_pts, dim_i, dim_j, args, rng,
                   label='Sublevel set', alpha=None):
    if not len(shell_pts):
        return
    pts = _maybe_subsample(shell_pts, args.shell_subsample, rng)
    ax.scatter(pts[:, dim_i], pts[:, dim_j],
               s=1, c='red',
               alpha=alpha if alpha is not None else args.shell_alpha,
               label=label, zorder=0)


def _plot_traj_line(ax, traj_arr, label, transient, zorder,
                    lw=0.8, alpha=0.7, color=None):
    if traj_arr is None:
        return
    ax.plot(traj_arr[transient:, 0], traj_arr[transient:, 1],
            color=color, lw=lw, alpha=alpha, label=label, zorder=zorder)


def _gather_panels(args, device):
    panels = []
    for title, pattern, proj_flag in VARIANTS:
        model_root = find_model_dir(args.root, pattern)
        model_dir = os.path.join(model_root, 'models')
        ckpt_path = os.path.join(model_dir, args.checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        print(f"\n=== {title} ===\n  dir: {model_root}")

        model, v_type = load_model(model_dir, args.checkpoint, device,
                                   proj_flag=proj_flag)
        print(f"  V type: {v_type}, proj_flag={proj_flag}")

        results_path = os.path.join(
            model_root, f'eval_{args.checkpoint.replace(".pt","")}_results.npz')
        cached = None if args.force_rollout else load_cached_trajs(results_path)
        if cached is not None:
            true = cached['true']
            pred = cached['pred']
            dt = cached.get('dt', None)
        else:
            data = np.load(args.data)
            test_data = data['test_data']
            dt = float(data['dt'])
            steps = args.rollout_steps or (test_data.shape[1] - 1)
            x0_all = test_data[:, 0, :]
            pred = rollout(model, x0_all, steps, device)
            true = test_data[:, :steps + 1, :]
        if dt is None:
            data = np.load(args.data)
            dt = float(data['dt'])
        transient_steps = min(int(args.transient_time / dt), true.shape[1] - 2)

        if proj_flag:
            shell_pts, vol_frac, n_in, c2 = sample_boundary_shell(
                model, device, args.n_samples, args.shell_frac, args.seed)
            print(f"  c^2 = {c2:.3f}, n_in_sublevel = {n_in:,} / {args.n_samples:,}"
                  f"  -> vol_frac = {vol_frac:.4f}")
            if args.shell_frac > 0:
                print(f"  shell pts (V in [{args.shell_frac:.2f}c^2, c^2]) = "
                      f"{len(shell_pts):,}")
            else:
                print(f"  sublevel pts (V <= c^2, no shell filter) = "
                      f"{len(shell_pts):,}")
        else:
            shell_pts = np.empty((0, 3), dtype=np.float32)
            vol_frac = float('nan')
            c2 = float((model.c ** 2).detach().cpu().item())
            print(f"  c^2 = {c2:.3f} (no projection -> sublevel set is not certified, "
                  f"skipping shell)")

        pts_t_xy = filter_finite(pooled_state_pairs(true, transient_steps, 0, 1))
        pts_p_xy = filter_finite(pooled_state_pairs(pred, transient_steps, 0, 1))
        pts_t_xz = filter_finite(pooled_state_pairs(true, transient_steps, 0, 2))
        pts_p_xz = filter_finite(pooled_state_pairs(pred, transient_steps, 0, 2))

        panels.append({
            'title': title,
            'v_type': v_type,
            'proj_flag': proj_flag,
            'c2': c2,
            'vol_frac': vol_frac,
            'shell_pts': shell_pts,
            'pts_t_xy': pts_t_xy,
            'pts_p_xy': pts_p_xy,
            'pts_t_xz': pts_t_xz,
            'pts_p_xz': pts_p_xz,
        })

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return panels


def _ensure_caches(args, device):
    for title, pattern, proj_flag in VARIANTS:
        model_root = find_model_dir(args.root, pattern)
        model_dir = os.path.join(model_root, 'models')
        compute_and_cache(
            model_dir, args.checkpoint, args.data, proj_flag, device,
            out_dir=model_root, rollout_steps=args.rollout_steps,
            n_pairs=args.n_pairs, mirror_T_total=args.mirror_T_total,
            mirror_transient=args.mirror_transient, mirror_ic_lim=args.mirror_ic_lim,
            mirror_seed=args.mirror_seed, force=args.force_rollout)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Plot the Moore-Spiegel V-parameterization comparison (paper Fig 6).")
    p.add_argument('--root', type=str, default='assets/models/moore_spiegel',
                   help='Parent dir containing the four variant subdirs.')
    p.add_argument('--data', type=str,
                   default='assets/data/moore_spiegel/dim3_M50_N20000_dt0p01_amp10p0_T6p0_R20p0.npz',
                   help='Moore-Spiegel test data npz.')
    p.add_argument('--out', type=str, default='figures/moore_spiegel',
                   help='Output directory for the figure and metrics.log.')
    p.add_argument('--checkpoint', type=str, default='E29999.pt')
    p.add_argument('--n_samples', type=int, default=500_000,
                   help='MC samples inside the bounding A-ellipsoid (sublevel scatter).')
    p.add_argument('--shell_frac', type=float, default=0.0,
                   help='Keep V in [shell_frac*c^2, c^2]; 0 = full sublevel set.')
    p.add_argument('--shell_alpha', type=float, default=0.25,
                   help='Per-point opacity for the sublevel-set scatter.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force_rollout', action='store_true',
                   help='Recompute the rollout + mirror caches even if present.')
    p.add_argument('--force_sampling', action='store_true',
                   help='Re-sample shell points even if <out>/levelset_data.npz matches.')
    p.add_argument('--transient_time', type=float, default=10.0)
    p.add_argument('--rollout_steps', type=int, default=None)
    p.add_argument('--hist_bins', type=int, default=80)
    p.add_argument('--shell_subsample', type=int, default=200_000,
                   help='Cap on sublevel-set points drawn per panel.')
    p.add_argument('--mixed_traj_map', type=str, default='0,0,0,0',
                   help='Per-variant traj_idx (VARIANTS order) for the figure traces.')
    p.add_argument('--mirror_traj_map', type=str, default='0,0,0,1',
                   help='Per-variant traj_idx for the density row (unconstrained '
                        'uses this test-data traj; constrained use the mirror cache).')
    p.add_argument('--density_xlim', type=str, default='-8,10')
    p.add_argument('--density_ylim', type=str, default='-10,10')

    p.add_argument('--n_pairs', type=int, default=3)
    p.add_argument('--mirror_T_total', type=float, default=200.0)
    p.add_argument('--mirror_transient', type=float, default=20.0)
    p.add_argument('--mirror_ic_lim', type=float, default=1.0)
    p.add_argument('--mirror_seed', type=int, default=7)
    args = p.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    print(f"[device] {device}")

    # Check if caches exist
    _ensure_caches(args, device)

    cache_path = os.path.join(args.out, 'levelset_data.npz')
    cache_meta = {
        'n_samples':  int(args.n_samples),
        'shell_frac': float(args.shell_frac),
        'seed':       int(args.seed),
        'checkpoint': str(args.checkpoint),
        'transient_time': float(args.transient_time),
    }
    cached_panels = (None if (args.force_sampling or args.force_rollout)
                     else panels_from_npz(cache_path, cache_meta))
    if cached_panels is not None:
        print(f"[cache] hit: loaded panel arrays from {cache_path}")
        panels = cached_panels
    else:
        panels = _gather_panels(args, device)
        panels_to_npz(panels, cache_path, cache_meta)
        print(f"[cache] wrote panel arrays to {cache_path}")

    gt_xy = [pn['pts_t_xy'] for pn in panels]
    gt_xz = [pn['pts_t_xz'] for pn in panels]
    shell_xy = [pn['shell_pts'][:, [0, 1]] for pn in panels]
    shell_xz = [pn['shell_pts'][:, [0, 2]] for pn in panels]
    lim_xy = [_range_of(gt_xy + shell_xy, 0), _range_of(gt_xy + shell_xy, 1)]
    lim_xz = [_range_of(gt_xz + shell_xz, 0), _range_of(gt_xz + shell_xz, 1)]
    fig_xlim, fig_ylim = lim_xy[0], lim_xy[1]

    for pn in panels:
        _, _, _, _, KL_xy, JS_xy = hist2d_kl_js(
            pn['pts_t_xy'], pn['pts_p_xy'], bins=args.hist_bins, lim=lim_xy)
        _, _, _, _, KL_xz, JS_xz = hist2d_kl_js(
            pn['pts_t_xz'], pn['pts_p_xz'], bins=args.hist_bins, lim=lim_xz)
        pn['KL_xy'] = KL_xy; pn['JS_xy'] = JS_xy
        pn['KL_xz'] = KL_xz; pn['JS_xz'] = JS_xz

    rng = np.random.default_rng(args.seed)

    # Produce Fig. 6 in the paper
    full_trajs = _load_full_trajs(args)
    mixed_traj_map = [int(s) for s in args.mixed_traj_map.split(',') if s.strip()]
    if len(mixed_traj_map) != len(panels):
        raise ValueError(f"--mixed_traj_map needs {len(panels)} indices")
    mirror_traj_map = [int(s) for s in args.mirror_traj_map.split(',') if s.strip()]
    if len(mirror_traj_map) != len(VARIANTS):
        raise ValueError(f"--mirror_traj_map needs {len(VARIANTS)} indices")
    per_variant_density, gt_pooled = _build_density_data(args, mirror_traj_map)
    combined_path = os.path.join(args.out, 'combined_levelset_density_x_xdot.png')
    _plot_combined_levelset_density(
        panels, full_trajs, mixed_traj_map, per_variant_density, gt_pooled,
        fig_xlim, fig_ylim, args, combined_path, rng,
        density_dim_pair=(0, 1), bins=args.hist_bins, color_scale='log')

    # Compute average KL divergences over 3 best trajectories that each model produces
    def _clean(s):
        return s.replace('$', '')
    data = np.load(args.data)
    test_data = data['test_data']
    dt_t = float(data['dt'])
    transient_search = min(int(args.transient_time / dt_t), test_data.shape[1] - 2)
    gt_all = test_data[:, transient_search:, :].reshape(-1, 3)
    gt_all = gt_all[np.isfinite(gt_all).all(axis=1)]

    def _shared_lim(di, dj):
        x_lo, x_hi = gt_all[:, di].min(), gt_all[:, di].max()
        y_lo, y_hi = gt_all[:, dj].min(), gt_all[:, dj].max()
        px, py = 0.05 * float(x_hi - x_lo), 0.05 * float(y_hi - y_lo)
        return [[float(x_lo - px), float(x_hi + px)], [float(y_lo - py), float(y_hi + py)]]
    sh_xy, sh_xz = _shared_lim(0, 1), _shared_lim(0, 2)
    n_test = test_data.shape[0]

    best_lines = [f'[Best-trajectory & averages] KL over {n_test} test '
                  f'trajectories per variant (shared GT-derived bins)']
    best_header = (f"  {'Variant':<24} {'best KL(x,xdot)':>16} {'idx':>5} "
                   f"{'top3 avg(x,xdot)':>17} {'avg KL(x,xdot)':>15}  "
                   f"{'best KL(x,xddot)':>17} {'idx':>5} "
                   f"{'top3 avg(x,xddot)':>18} {'avg KL(x,xddot)':>16}")
    best_lines += [best_header, '  ' + '-' * (len(best_header) - 2)]
    ckpt_stem = args.checkpoint.replace('.pt', '')

    def _top3(values, k=3):
        clean = sorted(v for v in values if not np.isnan(v))
        return float(np.mean(clean[:k])) if clean else float('nan')

    for title, pattern, proj_flag in VARIANTS:
        if not proj_flag:
            continue
        model_root = find_model_dir(args.root, pattern)
        npz = np.load(os.path.join(model_root, f'eval_{ckpt_stem}_results.npz'))
        true_arr = npz['true']; pred_arr = npz['pred']; npz.close()
        best_xy = (None, float('inf')); best_xz = (None, float('inf'))
        per_xy = []; per_xz = []
        for tidx in range(true_arr.shape[0]):
            tt_full = true_arr[tidx, transient_search:, :]
            pp_full = pred_arr[tidx, transient_search:, :]
            for (di, dj), lim, name in [((0, 1), sh_xy, 'xy'), ((0, 2), sh_xz, 'xz')]:
                tt = tt_full[:, [di, dj]]; pp = pp_full[:, [di, dj]]
                tt = tt[np.isfinite(tt).all(axis=1)]; pp = pp[np.isfinite(pp).all(axis=1)]
                if not len(tt) or not len(pp):
                    continue
                _, _, _, _, KL, _ = hist2d_kl_js(tt, pp, bins=args.hist_bins, lim=lim)
                if name == 'xy':
                    per_xy.append(KL)
                    if KL < best_xy[1]:
                        best_xy = (tidx, KL)
                else:
                    per_xz.append(KL)
                    if KL < best_xz[1]:
                        best_xz = (tidx, KL)
        avg_xy = float(np.nanmean(per_xy)) if per_xy else float('nan')
        avg_xz = float(np.nanmean(per_xz)) if per_xz else float('nan')
        best_lines.append(
            f"  {_clean(title):<24} {best_xy[1]:>16.4f} {best_xy[0]:>5} "
            f"{_top3(per_xy):>17.4f} {avg_xy:>15.4f}  "
            f"{best_xz[1]:>17.4f} {best_xz[0]:>5} "
            f"{_top3(per_xz):>18.4f} {avg_xz:>16.4f}")
    best_traj_summary = '\n'.join(best_lines)
    print('\n' + best_traj_summary)

    # print metrics summary table
    header = (f"{'Variant':<24} {'c^2':>8} {'vol_frac':>10} "
              f"{'KL(x,xdot)':>12} {'KL(x,xddot)':>13} "
              f"{'JS(x,xdot)':>12} {'JS(x,xddot)':>13}")
    sep = '=' * len(header)
    lines = ['V parameterization comparison - Moore-Spiegel', sep, header, sep]
    for pn in panels:
        lines.append(
            f"{pn['title']:<24} {pn['c2']:>8.2f} {pn['vol_frac']:>10.4e} "
            f"{pn['KL_xy']:>12.4e} {pn['KL_xz']:>13.4e} "
            f"{pn['JS_xy']:>12.4e} {pn['JS_xz']:>13.4e}")
    lines.append(sep)
    lines.append(f"Settings: n_samples={args.n_samples:,}, shell_frac={args.shell_frac}, "
                 f"hist_bins={args.hist_bins}, transient_time={args.transient_time}, "
                 f"seed={args.seed}")
    summary = '\n'.join(lines)
    print('\n' + summary)
    log_path = os.path.join(args.out, 'metrics.log')
    with open(log_path, 'w') as fp:
        fp.write(summary + '\n')
        fp.write('\n' + best_traj_summary + '\n')
    print(f"Saved metrics: {log_path}")


if __name__ == '__main__':
    main()

