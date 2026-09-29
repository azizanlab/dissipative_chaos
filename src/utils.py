import logging

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

import scipy.fft as fft
from scipy.fft import fft as _fft
from scipy.integrate import solve_ivp


class KuramotoSivashinskyODE:
    def __init__(self, M, L, nu, N_grid, dealias=True):
        self.M = M
        self.L = L
        self.nu = nu
        self.N_grid = N_grid
        self.dealias = dealias

        if self.N_grid < 3 * self.M:
            raise ValueError(f"N_grid >= 3*M required (got N_grid={N_grid}, M={M})")

        self.k = (2 * np.pi / self.L) * np.arange(1, self.M + 1)
        self.linear_op = self.k**2 - self.nu * self.k**4
        self.k_full = (2 * np.pi / self.L) * fft.fftfreq(self.N_grid, d=1.0/self.N_grid)

        self.u_hat_full = np.zeros(self.N_grid, dtype=np.complex128)
        self.u = np.zeros(self.N_grid, dtype=np.complex128)
        self.u2_hat = np.zeros(self.N_grid, dtype=np.complex128)
        self.nl_full = np.zeros(self.N_grid, dtype=np.complex128)

    def rhs(self, t, u_hat_M):
        self.u_hat_full.fill(0)
        self.u_hat_full[1:self.M+1] = u_hat_M
        self.u_hat_full[-self.M:] = np.conj(u_hat_M[::-1])

        self.u[:] = fft.ifft(self.u_hat_full, norm='ortho')
        self.u2_hat[:] = fft.fft(self.u*self.u, norm='ortho')

        if self.dealias:
            cutoff = self.N_grid // 3
            self.u2_hat[cutoff:-cutoff] = 0

        self.nl_full[:] = 0.5j * self.k_full * self.u2_hat

        return self.linear_op * u_hat_M - self.nl_full[1:self.M+1]

    def solve(self, u0_hat, t_end, dt_out, method='BDF', rtol=1e-7, atol=1e-8):
        logging.info(f"Integrating: M={self.M}, L={self.L}, nu={self.nu}, N={self.N_grid}, dealias={self.dealias}, method={method}")
        n_steps = int(round(t_end / dt_out))
        t_eval = np.linspace(0, t_end, n_steps + 1)
        sol = solve_ivp(self.rhs, [0, t_end], u0_hat,
                        method=method, t_eval=t_eval,
                        rtol=rtol, atol=atol)
        if not sol.success:
            logging.warning(f"solve_ivp failed: {sol.message}")
        return sol.t, sol.y

def tensor_lorenz63(X, sigma=10, beta=8/3, rho=28):
    assert(X.shape[1] == 3)
    dX = torch.zeros_like(X)
    dX[:, 0] = sigma * (X[:, 1] - X[:, 0])
    dX[:, 1] = X[:, 0] * (rho - X[:, 2]) - X[:, 1]
    dX[:, 2] = X[:, 0] * X[:, 1] - beta * X[:, 2]
    return dX

def tensor_moore_spiegel(X, T=6.0, R=20.0):
    assert(X.shape[1] == 3)
    dX = torch.zeros_like(X)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    dX[:, 0] = y
    dX[:, 1] = z
    dX[:, 2] = -z - (T - R + R * x**2) * y - T * x
    return dX

def tensor_lorenz96_5(X, F=8.0):
    assert(X.shape[1] == 5)
    dX = torch.zeros_like(X)
    dX[:, 0] = (X[:, 1] - X[:, 3]) * X[:, 4] - X[:, 0] + F
    dX[:, 1] = (X[:, 2] - X[:, 4]) * X[:, 0] - X[:, 1] + F
    dX[:, 2] = (X[:, 3] - X[:, 0]) * X[:, 1] - X[:, 2] + F
    dX[:, 3] = (X[:, 4] - X[:, 1]) * X[:, 2] - X[:, 3] + F
    dX[:, 4] = (X[:, 0] - X[:, 2]) * X[:, 3] - X[:, 4] + F
    return dX

def gen_real_multi_traj(M, N, dt, odefun=tensor_lorenz63, x_lim=50.0, x_dim=3, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X0 = (torch.rand(M, x_dim, device=device) * 2 * x_lim) - x_lim
    X = torch.zeros(M, N, x_dim, device=device)
    
    X[:, 0, :] = X0

    for i in tqdm(range(N - 1)):
        X[:, i + 1, :] = rk4_model(odefun, dt, y=X[:, i, :], **kwargs)
        if torch.isnan(X[:, i + 1, :]).any():
            raise ValueError(f"NaN encountered at time step {i+1}")
        
    return X

def gen_KS_data_BDF(
    num_traj: int,
    traj_len: int,
    dt: float,
    chaotic_system: KuramotoSivashinskyODE,
    x_lim: float,
    ic: str = 'physical',
    seed: int = None,
) -> np.ndarray:
    """
    For KS-ROM only, the numerical integration for generating ground truth trajectories requires better accuracy and stability than RK4.
    """
    M      = chaotic_system.M
    N_grid = chaotic_system.N_grid
    Y      = np.zeros((num_traj, traj_len, 2*M), dtype=float)

    for i in range(num_traj):
        if seed is not None:
            np.random.seed(seed + i)

        amp = x_lim
        if ic == 'fourier':
            # spectral IC
            u0_hat = amp * (np.random.randn(M) + 1j*np.random.randn(M))
        else:
            # physical-space IC
            u0_phys      = amp * np.random.randn(N_grid)
            u0_hat_full  = _fft(u0_phys, norm='ortho')
            u0_hat       = u0_hat_full[1:M+1]


        t, u_hat_t = chaotic_system.solve(
            u0_hat,
            t_end  = dt * (traj_len - 1),
            dt_out = dt,
            method = 'BDF'
        )

        data = np.hstack([u_hat_t.real.T, u_hat_t.imag.T])
        Y[i] = data[:traj_len]

    return Y


def rk4_model(func, dt, y, **kwargs):
    f1 = func(y, **kwargs)
    f2 = func(y + dt / 2 * f1, **kwargs)
    f3 = func(y + dt / 2 * f2, **kwargs)
    f4 = func(y + dt * f3, **kwargs)
    y1 = y + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return y1

class TrajectoryTensorDataset(Dataset):
    def __init__(self, trajectories, subtraj_length, stride):
        self.trajectories = trajectories
        self.subtraj_length = subtraj_length
        self.stride = stride
        self.indices = []
        
        num_traj, traj_length, _ = trajectories.shape
        for traj_index in range(num_traj):
            for start in range(0, traj_length - subtraj_length + 1, stride):
                self.indices.append((traj_index, start))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_index, start = self.indices[idx]
        return self.trajectories[traj_index, start:start + self.subtraj_length, :]