"""Generate training/test trajectory data for the four dissipative chaotic systems studied in the paper: Lorenz 63, Lorenz 96, KS-ROM, and Moore-Spiegel Oscillator.

Lorenz 63, Lorenz 96, and Moore-Spiegel are integrated using RK4. KS-ROM is integrated with solve_ivp(BDF) on the spectral reduced-order model.

Output is a compressed .npz with `train_data`, `test_data`, `dt`, and per-system metadata.
"""

import numpy as np
import torch
import os
import sys
import argparse

# --- set import paths for supporting files from other subdirectories ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("src", "eval"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -----------------------------------------------------------------------------

from utils import (
    gen_real_multi_traj,
    gen_KS_data_BDF,
    tensor_lorenz63,
    tensor_lorenz96_5,
    tensor_moore_spiegel,
    KuramotoSivashinskyODE,
)


def set_chaotic_system_params(syst_name, args):
    params = {}
    if syst_name == 'L63':
        params.update({
            'odefun':    tensor_lorenz63,
            'state_dim': 3,
            'data_xlim': 30,
            'train_M':   50,
            'train_N':   100,
            'test_M':    10,
            'test_N':    500,
            'discrete_dt': 0.01,
        })
    elif syst_name == 'L96':
        params.update({
            'odefun':    tensor_lorenz96_5,
            'state_dim': 5,
            'data_xlim': 10,
            'train_M':   50,
            'train_N':   100,
            'test_M':    10,
            'test_N':    500,
            'F':         8,
            'discrete_dt': 0.01,
        })
    elif syst_name == 'KS_ROM':
        M_dimension = 16
        L_domain    = 100.0
        nu_viscosity = args.nu if args.nu is not None else 4.0
        N_grid_default = 128

        chaotic_system = KuramotoSivashinskyODE(
            M = M_dimension,
            L = L_domain,
            nu= nu_viscosity,
            N_grid=N_grid_default
        )
        params.update({
            'chaotic_system': chaotic_system,
            'state_dim':      2*M_dimension,
            'train_M':        4,
            'train_N':        50000,
            'test_M':         5,
            'test_N':         300000,
            'discrete_dt':    0.01,
            'nu':            nu_viscosity,
        })
    elif syst_name == 'Moore_Spiegel':
        params.update({
            'odefun':    tensor_moore_spiegel,
            'state_dim': 3,
            'data_xlim': args.ms_xlim,
            'train_M':   50,
            'train_N':   20000,
            'test_M':    10,
            'test_N':    50000,
            'discrete_dt': 0.01,
            'T':         args.ms_T,
            'R':         args.ms_R,
        })
    else:
        raise ValueError(f"Unknown system: {syst_name}")

    return params


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--system', type=str, default='KS_ROM',
        choices=['L63', 'L96', 'KS_ROM', 'Moore_Spiegel']
    )

    # ── Moore-Spiegel oscillator ───────────────────────────────────
    p.add_argument('--ms_T', type=float, default=6.0)
    p.add_argument('--ms_R', type=float, default=20.0)
    p.add_argument('--ms_xlim', type=float, default=10.0)

    # ── KS-ROM ─────────────────────────────────────────────────
    p.add_argument('--ic', type=str, default='physical', choices=['physical', 'fourier'])
    p.add_argument('--seed', type=int, default=None,)
    p.add_argument('--amp', type=float, default=10.0)
    p.add_argument('--nu', type=float, default=None)

    args = p.parse_args()
    syst_name = args.system

    params = set_chaotic_system_params(syst_name, args)

    data_dir = os.path.join(os.getcwd(), 'Data', syst_name)
    os.makedirs(data_dir, exist_ok=True)

    if syst_name == 'KS_ROM':
        print(f"Generating data for KS-ROM with BDF solver…")
        Y_MN = gen_KS_data_BDF(
            num_traj      = params['train_M'],
            traj_len      = params['train_N'],
            dt            = params['discrete_dt'],
            chaotic_system = params['chaotic_system'],
            x_lim         = args.amp,
            ic            = args.ic,
            seed          = args.seed,
        )
        Y_MN_test = gen_KS_data_BDF(
            num_traj      = params['test_M'],
            traj_len      = params['test_N'],
            dt            = params['discrete_dt'],
            chaotic_system = params['chaotic_system'],
            x_lim         = args.amp,
            ic            = args.ic,
            seed          = args.seed,
        )
    else:
        print(f"Generating data for {syst_name} using RK4 solver…")
        # Seed the IC sampling (gen_real_multi_traj draws X0 via torch.rand) so
        # the staged Lorenz datasets are reproducible. train then test are drawn
        # from the same (seeded) stream, so they get distinct initial conditions.
        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        ode_kwargs = {}
        if 'F' in params:
            ode_kwargs['F'] = params['F']
        if syst_name == 'Moore_Spiegel':
            ode_kwargs['T'] = params['T']
            ode_kwargs['R'] = params['R']

        Y_MN = gen_real_multi_traj(
            params['train_M'], params['train_N'], params['discrete_dt'],
            odefun=params['odefun'],
            x_lim = params['data_xlim'],
            x_dim = params['state_dim'],
            **ode_kwargs
        )
        Y_MN_test = gen_real_multi_traj(
            params['test_M'], params['test_N'], params['discrete_dt'],
            odefun=params['odefun'],
            x_lim = params['data_xlim'],
            x_dim = params['state_dim'],
            **ode_kwargs
        )

    def _f2s(v):
        return str(v).replace('.', 'p')

    def _to_np(arr):
        return arr.detach().cpu().numpy() if hasattr(arr, 'detach') else np.asarray(arr)

    if syst_name == 'Moore_Spiegel':
        out_name = (
            f"dim{params['state_dim']}"
            f"_M{params['train_M']}_N{params['train_N']}"
            f"_dt{_f2s(params['discrete_dt'])}"
            f"_amp{_f2s(params['data_xlim'])}"
            f"_T{_f2s(params['T'])}"
            f"_R{_f2s(params['R'])}"
            ".npz"
        )
        out_path = os.path.join(data_dir, out_name)
        np.savez_compressed(
            out_path,
            train_data = _to_np(Y_MN),
            test_data  = _to_np(Y_MN_test),
            dt         = params['discrete_dt'],
            system     = syst_name,
            T          = params['T'],
            R          = params['R'],
        )
    elif syst_name == 'KS_ROM':
        out_name = (
            f"dim{params['state_dim']}"
            f"_M{params['train_M']}_N{params['train_N']}"
            f"_dt{_f2s(params['discrete_dt'])}"
            f"_amp{args.amp}"
            f"_nu{args.nu if args.nu is not None else '4p0'}"
            ".npz"
        )
        out_path = os.path.join(data_dir, out_name)
        np.savez_compressed(
            out_path,
            train_data = _to_np(Y_MN),
            test_data  = _to_np(Y_MN_test),
            dt         = params['discrete_dt'],
            system     = syst_name,
            ic_type    = args.ic,
            seed       = args.seed,
            nu         = params['nu'],
        )
    else:  # L63 / L96 — Lorenz systems (RK4; IC range = data_xlim, F for L96)
        fz = f"_F{params['F']}" if 'F' in params else ""
        out_name = (
            f"dim{params['state_dim']}"
            f"_M{params['train_M']}_N{params['train_N']}"
            f"_dt{_f2s(params['discrete_dt'])}"
            f"_xlim{_f2s(params['data_xlim'])}"
            f"{fz}.npz"
        )
        out_path = os.path.join(data_dir, out_name)
        np.savez_compressed(
            out_path,
            train_data = _to_np(Y_MN),
            test_data  = _to_np(Y_MN_test),
            dt         = params['discrete_dt'],
            system     = syst_name,
            data_xlim  = params['data_xlim'],
            F          = params.get('F'),
            seed       = args.seed,
        )

    print(f"\n Data for {syst_name} saved to {out_path}")
