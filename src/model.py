import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import rk4_model


class stable_lorenz_model(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.state_dim = params['state_dim']

        hf_dim = params['hf_dim']
        hV_dim = params['hV_dim']
        if params['f_activation'] == 'Tanh':
            self.f_activation = nn.Tanh()
        elif params['f_activation'] == 'ReLU':
            self.f_activation = nn.ReLU()
        elif params['f_activation'] == 'GeLU':
            self.f_activation = nn.GELU()
        else:
            raise RuntimeError('Unknown activation function: {}'.format(params['f_activation']))

        # Define fhat: MLP approximator for nonlinear dynamics.
        self.fhat = nn.Sequential(
            nn.Linear(self.state_dim, hf_dim),
            self.f_activation,
            nn.Linear(hf_dim, hf_dim),
            self.f_activation,
            nn.Linear(hf_dim, self.state_dim)
        )

        rehu_d = float(params["rehu_d"])
        if params['V_activation'] == 'Tanh':
            self.V_activation = nn.Tanh()
        elif params['V_activation'] == 'ReLU': 
            self.V_activation = nn.ReLU()
        elif params['V_activation'] == 'ReHU':
            self.V_activation = ReHU(rehu_d)

        # Building V dependent on the parameterization choice.
        V_in_dim = self.state_dim
        self.V_ellip = False
        self.V_quartic = False
        self.V_mlp = False
        if params["V_MLP"]:
            # MLP-PD construction + learnable ellipsoidal floor.
            if params["V_layer"] == "None":
                seq = nn.Sequential(
                    nn.Linear(V_in_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, 1)
                    )
            elif params["V_layer"] == "Tanh":
                seq = nn.Sequential(
                    nn.Linear(V_in_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, 1),
                    nn.Tanh(),
                    nn.Linear(1, 1, bias=False)
                    )
                V_scale = float(params["V_scale"]) if "V_scale" in params else 10.0
                seq[-1].weight.data = torch.tensor([[V_scale]], dtype=torch.float32)
                seq[-1].weight.requires_grad = False
            elif params["V_layer"] == "Sigmoid":
                seq = nn.Sequential(
                    nn.Linear(V_in_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, 1),
                    nn.Sigmoid(),
                    nn.Linear(1, 1, bias=False)
                    )
                V_scale = float(params["V_scale"]) if "V_scale" in params else 10.0
                seq[-1].weight.data = torch.tensor([[V_scale]], dtype=torch.float32)
                seq[-1].weight.requires_grad = False
            elif params["V_layer"] == "ReLU":
                seq = nn.Sequential(
                    nn.Linear(V_in_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, hV_dim),
                    self.V_activation,
                    nn.Linear(hV_dim, 1),
                    nn.ReLU()
                )
            self.V = V_psd_with_floor(seq, n=V_in_dim, d=rehu_d)
            self.V_mlp = True
        elif params["V_ellip"]:
            self.V = V_elliptical(self.state_dim)
            self.V_ellip = True
        elif params.get("V_quartic", False):
            self.V = V_quartic_sos(self.state_dim)
            self.V_quartic = True
        else:
            raise RuntimeError("Unknown V approximator")

        # projection tolerance eps for dVdx, avoids division by 0
        self.eps_proj = float(params["eps_proj"])
        self.proj_flag = params["proj_flag"]

        self.fix_c_flag = params['fix_c']

        if params['fix_c']:
            self.c = nn.Parameter(params["c_init"]*torch.ones(1), requires_grad=False)
        else:
            self.c = nn.Parameter(params["c_init"]*torch.ones(1), requires_grad=True)
            
        # Optional decay-rate coefficient in the projection (for the S-procedure): <grad V, f> + alpha^2 * (V - c^2) <= 0.
        # This is NOT used in any models in the paper, but the code supports learning alpha as well.
        alpha_init = float(params.get('alpha_init', 1.0))
        if alpha_init < 0:
            raise ValueError(f"alpha_init must be >= 0 (got {alpha_init})")
        self.fix_alpha_flag = bool(params.get('fix_alpha', False))
        self.alpha = nn.Parameter(np.sqrt(alpha_init)*torch.ones(1),
                                  requires_grad=not self.fix_alpha_flag)
        # Outdated parameters for testing purposes; not used in the paper.
        self.diagnose = params["diagnose"]
        self.discrete = params["discrete"]
        if params["discrete"]:
            self.dt = params["discrete_dt"]
            self.h = params["discrete_h"]
            self.N = int(self.dt/self.h)

    def get_ready(self, x):
        """Ensure x is on the correct device and is a leaf variable with requires_grad enabled.
        If x is a numpy array, create a new tensor.
        If x is already a tensor, avoid re‑creating it.
        """
        device = next(self.parameters()).device
        if not torch.is_tensor(x):
            # Create a new tensor from numpy.
            x = torch.tensor(x, dtype=torch.float32, device=device)
            x.requires_grad = True
        else:
            # If x is already a tensor, first ensure it is on the right device.
            x = x.to(dtype=torch.float32, device=device)
            # If x is not a leaf, detach and clone it to make it a leaf.
            if not x.is_leaf:
                x = x.detach().clone().requires_grad_(True)
            elif not x.requires_grad:
                x.requires_grad_()
        return x


    def f_proj(self, x):
        if not (isinstance(x, torch.Tensor) and x.requires_grad):
            x = self.get_ready(x)
        fhat = self.fhat(x)

        if self.V_ellip:
            # Analytic gradient: V = (x-x_0)^T Q (x-x_0)  ->  grad = 2 Q (x-x_0).
            V = self.V(x)
            Q = self.V.Q
            x_0 = self.V.x_0.squeeze(-1)
            V_grad = 2 * ((x - x_0) @ Q)
        elif self.V_quartic:
            # Analytic gradient from V_quartic_sos.grad.
            V = self.V(x)
            V_grad = self.V.grad(x)
        else:
            # MLP + PSD floor: needs autograd to compute gradients.
            with torch.enable_grad():
                x_g = x.detach().requires_grad_(True)
                V = self.V(x_g)
                V_grad = torch.autograd.grad(
                    [a for a in V], [x_g],
                    create_graph=True, only_inputs=True)[0]

        # Project the dynamics onto the dissipativity-certified half-space.
        # Constraint: <grad V, f> + (alpha^2 + 1e-3) * (V - c^2) <= 0
        # The 1e-3 floor keeps the effective decay rate strictly positive.
        alpha_eff = self.alpha**2 + 1e-3
        correction = V_grad * (F.relu((V_grad*fhat).sum(dim=1) + alpha_eff * (V[:,0] - self.c**2))/torch.clamp((V_grad**2).sum(dim=1), min=self.eps_proj))[:,None]
        f_proj = fhat - correction

        if self.diagnose:
            if torch.sum(V_grad * f_proj) + (V - self.c**2) > 0.:
                print("Projection is not working on x")

        return f_proj

    def _rhs(self, x):
        """RHS for RK4 integration of the autonomous learned dynamics."""
        return self.fhat(x)

    def forward(self, x):
        x = self.get_ready(x)
        fhat = self._rhs(x)
        V = self.V(x)

        if self.discrete == False:
            if self.proj_flag:
                f_proj = self.f_proj(x)
            else:
                f_proj = fhat

            return f_proj

        else:
            for i in range(self.N):
                if self.proj_flag:
                    x = rk4_model(self.f_proj, self.h, x)
                else:
                    x = rk4_model(self._rhs, self.h, x)
            return x

class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)

class V_elliptical(nn.Module):
    def __init__(self, n):
        super(V_elliptical, self).__init__()

        self.latent_dim = n
        # diagonal elements of the lower triangular matrix L
        self.log_diag_L = nn.Parameter(torch.zeros(self.latent_dim))

        # Learnable parameters for the lower triangular (off-diagonal) elements of L.
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        self.off_diag_L = nn.Parameter(torch.randn(len(tril_indices[0])) * 0.1)

        # Indices are not part of the parameters, so explicitly registered as a buffer.
        self.register_buffer('tril_indices', tril_indices)
    
        # Trainable vector x_0
        self.x_0 = nn.Parameter(torch.randn(n, 1))

        self.Q = None

    def _construct_Q(self):
        """
        Constructs the symmetric positive-definite matrix Q from L.
        """
        L = torch.zeros(self.latent_dim, self.latent_dim, device=self.log_diag_L.device)

        # Set the diagonal elements using the log_diag_L parameters.
        L.diagonal().copy_(torch.exp(self.log_diag_L))

        # Set the off-diagonal elements from the learned parameters.
        L[self.tril_indices[0], self.tril_indices[1]] = self.off_diag_L

        # Compute Q = L L^T
        Q = torch.matmul(L, L.T)
        return Q

        
    def forward(self, x):
        Q = self._construct_Q()

        self.Q = Q

        x_0 = self.x_0.squeeze(-1)
        diff = x - x_0
        
        # Calculate V for each input in the batch
        V = torch.einsum('bi,ij,bj->b', diff, Q, diff)
        V = V.unsqueeze(1)
        return V


# Helper functions for V_quartic_sos and V_psd_with_floor
def _cholesky_psd_param(n, init_off_scale=0.1):
    """Returns (log_diag_L, off_diag_L, tril_indices) for an n x n PD matrix
    parameterized as Q = L L^T, L lower-triangular with positive diagonal.
    Caller registers tril_indices as a buffer."""
    log_diag_L = nn.Parameter(torch.zeros(n))
    tril_indices = torch.tril_indices(row=n, col=n, offset=-1)
    off_diag_L = nn.Parameter(torch.randn(len(tril_indices[0])) * init_off_scale)
    return log_diag_L, off_diag_L, tril_indices


def _construct_chol(log_diag_L, off_diag_L, tril_indices, n):
    L = torch.zeros(n, n, device=log_diag_L.device, dtype=log_diag_L.dtype)
    L.diagonal().copy_(torch.exp(log_diag_L))
    L[tril_indices[0], tril_indices[1]] = off_diag_L
    return L @ L.T


class V_quartic_sos(nn.Module):
    """Sum-of-squares quartic Lyapunov function:
        V(x) = y^T A y + z_q(y)^T C z_q(y),    y = x - x_0
    with z_q(y) being the upper-triangular monomials of degree 2,
    z_q = [y_i y_j : i <= j] in R^{m_q}, m_q = n(n+1)/2.

    Analytical gradient:
        grad_V(y) = 2 A y + 2 J_q(y)^T C z_q(y)
    where J_q(y)[k, l] = d z_q[k] / d y_l.
    """
    def __init__(self, n):
        super().__init__()
        self.latent_dim = n
        self.m_q = n * (n + 1) // 2

        # Cholesky parameterization for A (n x n) and C (m_q x m_q).
        self.log_diag_LA, self.off_diag_LA, tril_A = _cholesky_psd_param(n)
        self.register_buffer('tril_A', tril_A)
        self.log_diag_LC, self.off_diag_LC, tril_C = _cholesky_psd_param(self.m_q)
        self.register_buffer('tril_C', tril_C)

        # Index tables for z_q and J_q.
        # idx_i, idx_j: for k in 0..m_q-1, the (i, j) pair with i <= j.
        ii, jj = [], []
        for i in range(n):
            for j in range(i, n):
                ii.append(i)
                jj.append(j)
        self.register_buffer('idx_i', torch.tensor(ii, dtype=torch.long))
        self.register_buffer('idx_j', torch.tensor(jj, dtype=torch.long))
        # E_i[k, l] = 1 if idx_i[k] == l, E_j similarly. Used to assemble J_q.
        E_i = torch.zeros(self.m_q, n)
        E_j = torch.zeros(self.m_q, n)
        for k in range(self.m_q):
            E_i[k, ii[k]] = 1.0
            E_j[k, jj[k]] = 1.0
        self.register_buffer('E_i', E_i)
        self.register_buffer('E_j', E_j)

        # Trainable center.
        self.x_0 = nn.Parameter(torch.randn(n, 1))

        self.Q = None

    def _construct_Q(self):
        return _construct_chol(self.log_diag_LA, self.off_diag_LA, self.tril_A, self.latent_dim)

    def _construct_C(self):
        return _construct_chol(self.log_diag_LC, self.off_diag_LC, self.tril_C,
                               self.m_q)

    def _z_q_and_J_q(self, y):
        z_q = y[:, self.idx_i] * y[:, self.idx_j]
        y_at_i = y[:, self.idx_i]
        y_at_j = y[:, self.idx_j]
        J_q = (y_at_j.unsqueeze(-1) * self.E_i.unsqueeze(0)
               + y_at_i.unsqueeze(-1) * self.E_j.unsqueeze(0))
        return z_q, J_q

    def forward(self, x):
        A = self._construct_Q()
        C = self._construct_C()
        self.Q = A
        x_0 = self.x_0.squeeze(-1)
        y = x - x_0
        z_q, _ = self._z_q_and_J_q(y)
        V_quad = torch.einsum('bi,ij,bj->b', y, A, y)
        V_quartic = torch.einsum('bk,kl,bl->b', z_q, C, z_q)
        return (V_quad + V_quartic).unsqueeze(1)

    def grad(self, x):
        """Analytical grad_x V(x), shape (B, n)."""
        A = self._construct_Q()
        C = self._construct_C()
        x_0 = self.x_0.squeeze(-1)
        y = x - x_0
        z_q, J_q = self._z_q_and_J_q(y)
        # 2 A y
        grad_quad = 2.0 * (y @ A)
        # 2 J_q^T (C z_q)
        Cz = z_q @ C
        grad_quartic = 2.0 * torch.einsum('bkl,bk->bl', J_q, Cz)
        return grad_quad + grad_quartic


class V_psd_with_floor(nn.Module):
    """MLP-based Lyapunov function with PSD construction and a learnable
    ellipsoidal floor:
        V(x) = ReHU( f(x) - f(x_0) ) + (x - x_0)^T Q (x - x_0)
    """
    def __init__(self, f, n, d=1.0):
        super().__init__()
        self.f = f
        self.latent_dim = n
        self.rehu = ReHU(d)
        self.log_diag_L, self.off_diag_L, tril_indices = _cholesky_psd_param(n)
        self.register_buffer('tril_indices', tril_indices)
        self.x_0 = nn.Parameter(torch.randn(n, 1))
        self.Q = None

    def _construct_Q(self):
        return _construct_chol(self.log_diag_L, self.off_diag_L,
                               self.tril_indices, self.latent_dim)

    def forward(self, x):
        Q = self._construct_Q()
        self.Q = Q
        x_0 = self.x_0.squeeze(-1)
        diff = x - x_0
        quad = torch.einsum('bi,ij,bj->b', diff, Q, diff).unsqueeze(1)
        bump = self.rehu(self.f(x) - self.f(x_0.unsqueeze(0)))
        return bump + quad