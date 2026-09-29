import argparse
import os
import sys

# --- repo path setup: make src/, eval/ importable ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("src", "eval"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -----------------------------------------------------------------------------

import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from utils import (TrajectoryTensorDataset, gen_real_multi_traj, gen_KS_data_BDF,
                   rk4_model, tensor_lorenz63, tensor_lorenz96_5, KuramotoSivashinskyODE)

from model import stable_lorenz_model
import torch
import torch.nn.functional as F
from torch import nn

from tqdm import tqdm
import pickle

import logging
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams

plt.rcParams.update({'font.size': 20})

def spectral_norm(module):
    total_spectral_norm = 0
    for layer in module:
        if isinstance(layer, nn.Linear):
            u, s, v = torch.svd(layer.weight)
            total_spectral_norm += torch.max(s)
    return total_spectral_norm

def l2_norm(module):
    total_l2_norm = 0
    for param in module.parameters():
        total_l2_norm = torch.sum(param ** 2)
    return total_l2_norm

def ellip_vol(model):
    Q = model.V._construct_Q()

    det_Q = torch.linalg.det(Q)
    n = model.state_dim
    # volume proxy, avoiding the gamma function and other constant factors since we only care about relative volumes
    vol = torch.sqrt((model.c**2) / det_Q)
    vol = np.pi ** (n/2) * vol

    return vol


def scaled_ellip_vol(model):
    """Log-det form of the certified-volume proxy, used for KS-ROM (state_dim=32), used to improve training stability. """
    d = model.V.log_diag_L.numel()
    c_val = model.c ** 2
    log_det_Q = 2 * torch.sum(model.V.log_diag_L)
    det_factor = torch.exp(- 1/2 * log_det_Q)
    if model.fix_c_flag:
        vol = det_factor
    else:
        vol = (c_val ** (d / 2)) * det_factor
    return vol


def sample_q_ellipsoid(model, n_samples, device):
    """Sample n_samples points uniformly inside the ellipsoid defined by the current Q. Helpful for the Monte-Carlo frac_smooth used in MLP-PD parameterization training process.
    """
    with torch.no_grad():
        Q = model.V._construct_Q()
        x_0 = model.V.x_0.squeeze(-1)
        c2 = model.c ** 2
        n = x_0.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(Q)
        L = eigvecs * (1.0 / torch.sqrt(eigvals.clamp_min(1e-12)))[None, :]
        gauss = torch.randn(n_samples, n, device=device)
        gauss = gauss / gauss.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = torch.rand(n_samples, 1, device=device) ** (1.0 / n)
        u = gauss * r
        y = torch.sqrt(c2) * (u @ L.T)
        x_mc = y + x_0
    return x_mc

def train_epoch(model, data_loader, loss_fun, optimizer, params, device):
    model.to(device)

    model.train()
    total_loss = 0.0
    total_dynamic_loss = 0.0
    total_reg_loss = 0.0
    num_batch = len(data_loader)

    for true_trajectory in data_loader:
        model.zero_grad()
        initial_state = true_trajectory[:, 0, :]
        traj_length = true_trajectory.shape[1]
        dynamic_loss = torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)
        current_state = initial_state.to(device)

        for step_id in range(traj_length-1):
            next_true_state = true_trajectory[:, step_id+1, :].to(device)
            predicted_state = model(current_state)
            dynamic_loss += loss_fun(predicted_state, next_true_state)
            current_state = predicted_state.detach()

        dynamic_loss = dynamic_loss / (traj_length-1)

        # Volume regularizer. 
        if params['reg_vol']:
            if params.get('system') == 'KS_ROM':
                vol = scaled_ellip_vol(model)
                reg_loss += params['lam_reg_vol'] * torch.sum(vol)
            else:
                vol = ellip_vol(model)
                reg_loss += params['lam_reg_vol'] * vol.squeeze()

        # Monte-Carlo Q-ellipsoid losses (used by the MLP-PD certificate).
        need_mc = (params['lam_frac'] > 0 or params['lam_bump'] > 0)
        if need_mc:
            x_mc = sample_q_ellipsoid(model, params['mc_n'], device)
            V_mc = model.V(x_mc).squeeze(-1)
            c2 = (model.c ** 2).squeeze()

            # Additive frac penalty: frac_smooth = (1/N) sum sigma((c^2-V)/tau), which is an estimate of Vol({V<=c^2}) / Vol(Q-ellipsoid).
            if params['lam_frac'] > 0:
                frac_smooth = torch.sigmoid((c2 - V_mc) / params['mc_tau']).mean()
                reg_loss = reg_loss + params['lam_frac'] * frac_smooth

            # Bump anti-collapse (MLP-PD only): Encourage MLP components to take larger values
            if params['lam_bump'] > 0 and params['V_MLP']:
                x_0_v = model.V.x_0.squeeze(-1)
                bump_arg = (model.V.f(x_mc) - model.V.f(x_0_v.unsqueeze(0))).squeeze(-1)
                d_rehu = float(params['rehu_d'])
                loss_bump = F.softplus(d_rehu - bump_arg).mean()
                reg_loss = reg_loss + params['lam_bump'] * loss_bump

        # V-on-data anchor loss (MLP-PD only): Prevent V to be lifted everywhere when lam_frac is active, as we observe failcase where V takes very large value to lower frac_smooth loss
        if params['lam_anchor'] > 0:
            x_data = true_trajectory.reshape(-1, true_trajectory.shape[-1]).to(device)
            V_data = model.V(x_data).squeeze(-1)
            c2 = (model.c ** 2).squeeze()
            V_target = params['V_anchor_frac'] * c2
            loss_anchor = F.relu(V_data - V_target).mean()
            reg_loss = reg_loss + params['lam_anchor'] * loss_anchor

        loss = params.get('dyn_weight', 1.0) * dynamic_loss + reg_loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dynamic_loss += dynamic_loss.item()
        total_reg_loss += reg_loss.item()

    avg_loss = total_loss/num_batch
    avg_dynamic_loss = total_dynamic_loss/num_batch
    avg_reg_loss = total_reg_loss/num_batch
    return avg_loss, avg_dynamic_loss, avg_reg_loss
        

def train(params):
    model = stable_lorenz_model(params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("is_available =", torch.cuda.is_available())  
    print("device_count =", torch.cuda.device_count())  
    print("device name =", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "None")

    logging.info("Training data sampled from multiple trajectories")

    # Generate training data if needed
    if not params['load_data']:
        if params['system'] == 'KS_ROM':
            Y_MN = gen_KS_data_BDF(
                num_traj=params['train_M'], traj_len=params['train_N'],
                dt=params['discrete_dt'], chaotic_system=params['chaotic_system'],
                x_lim=params['ic_amp'], ic=params['ic'], seed=params['seed'])
            Y_MN = torch.tensor(np.asarray(Y_MN, dtype=np.float32))
        elif params['odefun'] == tensor_lorenz96_5:
            Y_MN = gen_real_multi_traj(params['train_M'], params['train_N'], params['discrete_dt'], odefun=params['odefun'], x_lim=params['data_xlim'], x_dim=params['state_dim'], F=params['F'])
        else:
            Y_MN = gen_real_multi_traj(params['train_M'], params['train_N'], params['discrete_dt'], odefun=params['odefun'], x_lim=params['data_xlim'], x_dim=params['state_dim'])
    # Load data if possible
    else:
        if params['data_loc'].endswith('.npz'):
            data = np.load(params['data_loc'])
            Y_MN = torch.tensor(data['train_data'][:params['train_M'], :params['train_N'], :],
                                dtype=torch.float32)
        else:
            with open(params['data_loc'], 'rb') as f:
                Y_MN = pickle.load(f)
    

    Traj_data = TrajectoryTensorDataset(Y_MN, subtraj_length=params['train_sub_N'], stride=params['stride'])
    Traj_loader = DataLoader(Traj_data, batch_size=params['batch_size'], shuffle=True)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    num_epochs = params['epochs']
    total_loss_log = np.zeros(num_epochs)
    dynamic_loss_log = np.zeros(num_epochs)
    reg_loss_log = np.zeros(num_epochs)
    c_log = np.zeros(num_epochs)

    plot_directory = f"{params['model_dir']}/plots"
    if not os.path.exists(plot_directory):
        try:
            os.makedirs(plot_directory, exist_ok=True)
        except FileExistsError:
            print(f"The directory {plot_directory} already exists.")

    model_directory = f"{params['model_dir']}/models"
    if not os.path.exists(model_directory):
        try:
            os.makedirs(model_directory, exist_ok=True)
        except FileExistsError:
            print(f"The directory {model_directory} already exists.")
    
    for epoch_id in tqdm(range(num_epochs)):
        total_loss_log[epoch_id], dynamic_loss_log[epoch_id], reg_loss_log[epoch_id] = train_epoch(model, Traj_loader, loss_function, optimizer, params, device)
        c_log[epoch_id] = float(model.c.detach().cpu())**2
        if epoch_id % params["log_freq"] == 0 or epoch_id == num_epochs-1:
            logging.info(f"Epoch {epoch_id}, total_loss: {total_loss_log[epoch_id]}, Dynamic_loss: {dynamic_loss_log[epoch_id]}, Reg_loss: {reg_loss_log[epoch_id]}, c: {model.c}\n-----------")

            model_save_name = f'E{epoch_id}.pt'
            torch.save(model.state_dict(), model_directory + '/' + model_save_name)
    
    fig_size = (6, 6)
    fig1, ax1 = plt.subplots(1, 1, figsize=fig_size)
    ax1.plot(range(num_epochs), total_loss_log)
    ax1.plot(range(num_epochs), dynamic_loss_log)
    ax1.plot(range(num_epochs), reg_loss_log)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(["Total", "Dynamic", "Reg"])
    fig1.savefig(plot_directory + '/loss.png')

    fig2, ax2 = plt.subplots(1, 1, figsize=fig_size)
    ax2.plot(range(num_epochs), c_log)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(r"$c^2$")
    fig2.savefig(plot_directory + '/c.png')

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', '--epochs', dest='num_epochs', type=int, help='specify number of epochs', default=1500)
    parser.add_argument('--lr', type=float, help='specify learning rate', default=5e-4)
    parser.add_argument('--batch_size', type=int, help='specify batch size', default=1024)
    parser.add_argument('--seed', type=int, help='set random initialization seed', default=0)
    parser.add_argument('--system', type=str, help='specify system',
                        choices=['L63', 'L96_5', 'KS_ROM', 'Moore_Spiegel'], default='L63')
    parser.add_argument('--dyn_weight', type=float, default=1.0,
                        help='weight on the dynamic (prediction) loss')
    parser.add_argument('--hf_dim', type=int, help='specify fhat hidden layer size', default=64)
    parser.add_argument('--hV_dim', type=int, help='specify Vnet hidden layer size', default=64)
    parser.add_argument('--f_activation', type=str, help='specify activation function', default='GeLU')
    parser.add_argument('--V_activation', type=str, help='specify activation function', default='ReLU')
    parser.add_argument('--V_layer', type=str, help='specify additional layer attached to g_V', default='None')
    parser.add_argument('--V_scale', type=float, help='specify scale for Vnet output', default=10.0)
    parser.add_argument('--save_model', action='store_true', help='save model')
    parser.add_argument('--train_M', type=int, help='set number of trajectories', default=20)
    parser.add_argument('--train_N', type=int, help='set trajectory length', default=1000)
    parser.add_argument('--train_sub_N', type=int, help='set subtrajectory length', default=100)
    parser.add_argument('--stride', type=int, help='set stride', default=100)
    parser.add_argument('--x_lim', type=float, help='set x limit', default=50)
    parser.add_argument('--c_init', type=float, help='set initial c', default=10.0)
    parser.add_argument('--fix_c', action='store_true', help='fix c')
    parser.add_argument('--alpha_init', type=float, default=1.0,
                        help='initial *effective* decay coefficient (alpha^2) in '
                             '<grad V, f> + (alpha^2 + 1e-3)*(V - c^2) <= 0; '
                             'stored param is sqrt(alpha_init)')
    parser.add_argument('--fix_alpha', action='store_true',
                        help='fix alpha at sqrt(alpha_init) (otherwise learnable)')
    parser.add_argument('--proj_flag', action='store_true', help='set projection flag')
    parser.add_argument('--eps_proj', type=float, help='set projection tolerance', default=1e-4)
    parser.add_argument('--rehu_d', type=float, help='set ReHU d', default=0.01)
    parser.add_argument('--diagnose', action='store_true', help='diagnose')
    parser.add_argument('--log_freq', type=int, help='set logging frequency', default=5000)
    parser.add_argument('--discrete', action='store_true', help='discrete dynamics')
    parser.add_argument('--discrete_dt', type=float, help='discrete time step', default=0.01)
    parser.add_argument('--discrete_h', type=float, help='discrete steps', default=0.01)
    parser.add_argument('--V_MLP', action='store_true',
                        help='use MLP+PSD with ellipsoidal floor (V_psd_with_floor)')
    parser.add_argument('--V_ellip', action='store_true', help='use ellipsoidal Vnet')
    parser.add_argument('--V_quartic', action='store_true',
                        help='use no-cross quartic SOS Vnet (V = y^T A y + z_q^T C z_q)')
    parser.add_argument('--reg_vol', action='store_true',
                        help='regularize the certified volume via the analytic '
                             'ellip_vol(Q, c) proxy (used by all final models)')
    parser.add_argument('--lam_reg_vol', type=float, help='lambda for volume regularization', default=1e-2)
    parser.add_argument('--lam_frac', type=float, default=0.0,
                        help='additive Monte-Carlo frac_smooth penalty weight (MLP-PD); 0 disables')
    parser.add_argument('--lam_bump', type=float, default=0.0,
                        help='bump anti-collapse penalty weight (V_MLP only): '
                             'softplus(d - (f(x)-f(x_0))) on Q-ellipsoid samples; 0 disables')
    parser.add_argument('--mc_n', type=int, default=256,
                        help='number of Monte-Carlo samples per step (shared across MC losses)')
    parser.add_argument('--mc_tau', type=float, default=2.0,
                        help='temperature for sigmoid surrogate of frac (typical: 1-5%% of c^2)')
    parser.add_argument('--lam_anchor', type=float, default=0.0,
                        help='V-on-data anchor penalty weight: ReLU(V(x_data) - V_anchor_frac*c^2). '
                             'Caps V on observed trajectory data to prevent the "lift V everywhere" '
                             'pathology when lam_frac is active. 0 disables.')
    parser.add_argument('--V_anchor_frac', type=float, default=0.5,
                        help='V_target = V_anchor_frac * c^2 for the anchor loss (default 0.5)')
    parser.add_argument('--save_dir', type=str, help='set save directory', default='Experiment_results')
    parser.add_argument('--L96_F', type=float, help='set F for L96', default=8.0)
    # KS-ROM
    parser.add_argument('--rom_M', type=int, default=16, help='KS-ROM: retained Fourier modes')
    parser.add_argument('--rom_L', type=float, default=100.0, help='KS-ROM: domain length')
    parser.add_argument('--rom_nu', type=float, default=4.0, help='KS-ROM: viscosity')
    parser.add_argument('--rom_N_grid', type=int, default=128, help='KS-ROM: collocation points')
    parser.add_argument('--ic', type=str, choices=['physical', 'fourier'], default='physical',
                        help='KS-ROM: random initial-condition type for on-the-fly generation')
    parser.add_argument('--ic_amp', type=float, default=1.0,
                        help='KS-ROM: initial-condition amplitude for on-the-fly generation')
    parser.add_argument('--load_data', action='store_true', help='load trajectory data')
    parser.add_argument('--data_loc', type=str, help='set load directory', default=None)

    args = parser.parse_args()
    print(f"Projection: {args.proj_flag}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set initial state
    params = {
        'epochs': args.num_epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'hf_dim': args.hf_dim,
        'hV_dim': args.hV_dim,
        'f_activation': args.f_activation,
        'V_activation': args.V_activation,
        'V_layer': args.V_layer,
        'V_scale': args.V_scale,
        'save_model': args.save_model,
        'train_M': args.train_M,
        'train_N': args.train_N,
        "train_sub_N": args.train_sub_N,
        "stride": args.stride,
        'data_xlim': args.x_lim,
        'c_init': args.c_init,
        'fix_c': args.fix_c,
        'alpha_init': args.alpha_init,
        'fix_alpha': args.fix_alpha,
        'proj_flag': args.proj_flag,
        'eps_proj': args.eps_proj,
        'rehu_d': args.rehu_d,
        'diagnose': args.diagnose,
        'log_freq': args.log_freq,
        'discrete': args.discrete,
        'discrete_dt': args.discrete_dt,
        'discrete_h': args.discrete_h,
        'V_MLP': args.V_MLP,
        'V_ellip': args.V_ellip,
        'V_quartic': args.V_quartic,
        'reg_vol': args.reg_vol,
        'lam_reg_vol': args.lam_reg_vol,
        'lam_frac': args.lam_frac,
        'lam_bump': args.lam_bump,
        'mc_n': args.mc_n,
        'mc_tau': args.mc_tau,
        'lam_anchor': args.lam_anchor,
        'V_anchor_frac': args.V_anchor_frac,
        'load_data': args.load_data,
        'data_loc': args.data_loc,
        'system': args.system,
        'dyn_weight': args.dyn_weight,
        'ic': args.ic,
        'ic_amp': args.ic_amp,
    }

    if args.system == 'L63':
        params['state_dim'] = 3
        params['odefun'] = tensor_lorenz63
    elif args.system == 'L96_5':
        params['state_dim'] = 5
        params['odefun'] = tensor_lorenz96_5
        params['F'] = args.L96_F
    elif args.system == 'Moore_Spiegel':
        params['state_dim'] = 3
        params['odefun'] = None
        params['load_data'] = True
        params['data_loc'] = args.data_loc
    elif args.system == 'KS_ROM':
        params['chaotic_system'] = KuramotoSivashinskyODE(
            M=args.rom_M, L=args.rom_L, nu=args.rom_nu,
            N_grid=args.rom_N_grid, dealias=True)
        params['state_dim'] = 2 * args.rom_M 
        params['odefun'] = None
    else:
        raise ValueError("System not implemented (choose L63, L96_5, KS_ROM, or Moore_Spiegel)")

    reg_name = ''
    if params['reg_vol']:
        reg_name += 'vol'
    if params['lam_frac'] > 0:
        reg_name += f'_frac{params["lam_frac"]}_tau{params["mc_tau"]}'
    if params['lam_bump'] > 0:
        reg_name += f'_bump{params["lam_bump"]}'
    if params['lam_anchor'] > 0:
        reg_name += f'_anch{params["lam_anchor"]}f{params["V_anchor_frac"]}'

    if params['V_ellip']:
        v_tag = 'Vellip'
    elif params['V_quartic']:
        v_tag = 'Vquartic'
    elif params['V_MLP']:
        v_tag = 'VmlpPSD'
    else:
        v_tag = 'V?'

    log_save_name = f'{v_tag}_E{args.num_epochs}_LR{params["lr"]}_multitraj_M_{params["train_M"]}_N_{params["train_N"]}_subN_{params["train_sub_N"]}_sride_{params["stride"]}_proj_{params["proj_flag"]}_c0_{params["c_init"]}_fixc_{params["fix_c"]}_lam_{params["lam_reg_vol"]}_Reg_{reg_name}_a0_{params["alpha_init"]}_fixa_{params["fix_alpha"]}'

    save_dir = args.system + '/' + args.save_dir

    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            print(f"The directory {save_dir} already exists.")

    model_dir = f"{save_dir}/{log_save_name}"
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
        except FileExistsError:
            print(f"The directory {model_dir} already exists.")
    
    params['model_dir'] = model_dir

    logging.basicConfig(filename=f"{model_dir}/loss_info.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    model = train(params)
    logging.shutdown()
