import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import Scaler, beta_calculator, desired_ratio_var_H_overlap, desired_var_0_mu_H_overlap, unconcatenate_mu, unconcatenate_var, create_scalers, scale_data
from data_generation import generate_pretrain_lofi_data, generate_lofi_data, generate_hifi_lofi_data, generate_test_data
import mlflow
import pandas as pd
import matplotlib.ticker as ticker
import os

scaled = True
device = 'cpu'
torch.manual_seed(2025)
np.random.seed(2025)

# Set plotting parameters
title_fontsize = 12
axis_fontsize = 14
legend_fontsize = 9
other_linewidth = 1.5
linewidth = 1
scatter_size = 0.75

def function(x):
    return torch.sin(3 * np.pi * x) + 0.5 * x + 1 / 4
def lofi_function(x):
    return torch.sin(3 * np.pi * x)

## Data Generation
pre_train_lofi_regions = [(0, 1)]  # Domain of interest for plotting test
pre_train_lofi_num_points = [1000]
pre_train_lofi_total_num_points = sum(pre_train_lofi_num_points)

# Set regions where its only Lofi data
lofi_regions = [(0.5, 1)]
lofi_regions_num_points = [100]
lofi_total_num_points = sum(lofi_regions_num_points)

hifi_regions = [(0, 0.5)]
hifi_regions_num_points = [100]
hifi_total_num_points = sum(hifi_regions_num_points)
hifi_regions_noise_stds = [0.05]  # Small amounts of noise
hifi_regions_noise_mu = [0]

valid_regions = (0, 1)
valid_regions_num_points = 100

test_regions = (0, 1)
test_regions_num_points = 1000

# Create data
x_lofi_pretrain_data, y_lofi_pretrain_data = generate_pretrain_lofi_data(pre_train_lofi_total_num_points, pre_train_lofi_regions, pre_train_lofi_num_points, lofi_function)
x_lofi_only_data, y_lofi_only_data = generate_lofi_data(lofi_total_num_points, lofi_regions, lofi_regions_num_points, lofi_function)
x_hifi_data, y_hifi_data, x_lofi_match_data, y_lofi_match_data = generate_hifi_lofi_data(hifi_total_num_points, hifi_regions, hifi_regions_num_points, hifi_regions_noise_mu, hifi_regions_noise_stds, function, lofi_function)
x_valid_data, y_valid_data = generate_test_data(valid_regions, valid_regions_num_points, function)
x_test_data, y_test_data = generate_test_data(test_regions, test_regions_num_points, function)

# Concatenate lofi match x and y data before placing into []
x_lofi_data = torch.cat((x_lofi_match_data, x_lofi_only_data), 0)
y_lofi_data = torch.cat((y_lofi_match_data, y_lofi_only_data), 0)

# Create scalers
scalers = create_scalers(x_lofi_pretrain_data, y_lofi_pretrain_data)
x_Global_Scaler, y_Global_Scaler = scalers

# Scale data
x_hifi_scaled, y_hifi_scaled = scale_data(x_hifi_data, y_hifi_data, scalers)
x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
x_lofi_pretrain_scaled, y_lofi_pretrain_scaled = scale_data(x_lofi_pretrain_data, y_lofi_pretrain_data, scalers)
x_valid_scaled, y_valid_scaled = scale_data(x_valid_data, y_valid_data, scalers, label=False)
x_test_data_scaled, y_test_data_scaled = scale_data(x_test_data, y_test_data, scalers, label=False)

# Concatenate and store certain data sets together
x_H_train_data = torch.cat((x_hifi_scaled, x_lofi_scaled), 0)
y_H_train_data = torch.cat((y_hifi_scaled, y_lofi_scaled), 0)

# Create all training sets
H_train_data = [x_H_train_data.to(device), y_H_train_data.to(device)]
L_train_data = [x_lofi_scaled.to(device), y_lofi_scaled.to(device)]
D_train_data = [x_hifi_scaled.to(device), y_hifi_scaled.to(device)]

hifi_pretrain_data = [x_hifi_scaled.to(device), y_hifi_scaled.to(device)]
lofi_pretrain_data = [x_lofi_pretrain_scaled.to(device), y_lofi_pretrain_scaled.to(device)]
valid_data = [x_valid_scaled.to(device), y_valid_scaled.to(device)]

# Prior Hyper Parameter = 0  # LEGACY NEVER CHANGED FROM 0
n_lofi = 1  # NEVER CHANGED FROM 1
alpha_0 = 1  # Can set this to whatever we want.
var_H = 0.00001  # This is the max value for the Variance on the H value.
var_H_scaled = var_H / (y_Global_Scaler.std ** 2)  # matches training script: scaled ONCE

# Same sweep values as the training script
mode_var_h_list = [1, 0.1, 0.01, 0.001]#, 0.0001]#, 0.00001]
r_store = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]

# Precompute n_H_var / var_0 for every r (identical for every mode_var_h, matches training script)
var_0_store = []
n_H_store = []
for r in r_store:
    r_mu = r
    r_var = r_mu
    n_H_var = desired_ratio_var_H_overlap(r_var, alpha_0)
    var_0 = desired_var_0_mu_H_overlap(r_mu, n_H_var, var_H_scaled)
    var_0_store.append(var_0)
    n_H_store.append(n_H_var)

def gaussian_nll(input, target, var, eps=1e-6, reduction='mean'):
    var = var.clamp(min=eps)
    loss = 0.5 * (torch.log(var) + (input - target).pow(2) / var)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def evaluate(model, start=0, end=1, steps=1000, scaled=True):
    """Returns mse, avg sigma^2, and NLL over the given x-range, unscaled to true target space."""
    plot_model = model.to('cpu')
    plot_model.eval()

    x_test = torch.linspace(start, end, steps)
    y_test = torch.tensor(function(x_test))

    if scaled:
        x_Global_Scaler = scalers[0]
        y_Global_Scaler = scalers[1]
        x_test_input = x_Global_Scaler.transform(x_test)
    else:
        x_test_input = x_test

    mu_test, var_test = plot_model(x_test_input.reshape(x_test_input.shape[-1], -1))
    mu_test = mu_test.detach()
    sigma_test = var_test.detach()
    mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
    var_test_H, var_test_L = unconcatenate_var(sigma_test)

    if scaled:
        mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
        var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)

    mse_H = torch.sum((mu_test_H.flatten() - y_test.flatten().squeeze().numpy()) ** 2) / len(mu_test_H.flatten())
    var_H_avg = np.mean(var_test_H.numpy())
    nll_H = gaussian_nll(mu_test_H, y_test, var_test_H)

    return mse_H.item(), var_H_avg, nll_H.item()


def plot_results(model, title, start=0, end=1, steps=1000, scaled=True, show_data=False, directory=None):
    """Same visual style as your single-r plotting script, re-used per (mode_var_h, r) combination."""
    plot_model = model.to('cpu')
    plot_model.eval()

    x_test = torch.linspace(start, end, steps)
    y_test = torch.tensor(function(x_test))

    if scaled:
        x_Global_Scaler = scalers[0]
        y_Global_Scaler = scalers[1]
        x_test_input = x_Global_Scaler.transform(x_test)
    else:
        x_test_input = x_test

    mu_test, var_test = plot_model(x_test_input.reshape(x_test_input.shape[-1], -1))
    mu_test = mu_test.detach()
    sigma_test = var_test.detach()
    mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
    var_test_H, var_test_L = unconcatenate_var(sigma_test)

    if scaled:
        mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
        mu_test_L = y_Global_Scaler.inverse_transform(mu_test_L)
        mu_test_D = y_Global_Scaler.inverse_transform_mu_D(mu_test_D)
        var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)

    sig_test_H = np.sqrt(var_test_H.numpy())

    fig, ax = plt.subplots(dpi=500)
    ax.fill_between(x_test, mu_test_H - 2 * sig_test_H, mu_test_H + 2 * sig_test_H,
                     label='$\\hat\\mu_{Y_H}(x; \\theta_{\\mu_{Y_H}}) \\pm 2\\sqrt{\\hat\\sigma^2_{Y_H}(x;\\theta_{\\sigma^2_{Y_H}})}$',
                     color='green', alpha=0.1, zorder=3)
    ax.plot(x_test, mu_test_L + mu_test_D,
            label='$\\hat\\mu_{Y_L}(x; \\theta_{\\mu_{Y_L}})$ + $\\hat\\mu_{D}(x, \\hat\\mu_{Y_L}(x;\\theta_{\\mu_{Y_L}}); \\theta_{\\mu_{D}})$',
            color='purple', alpha=0.5, zorder=3, linewidth=other_linewidth + 2, linestyle=':')
    ax.plot(x_test, y_test, label='Ground Truth', color='orange', zorder=3, linewidth=other_linewidth)
    ax.plot(x_test, mu_test_H, label='$\\hat\\mu_{Y_H}(x; \\theta_{\\mu_{Y_H}})$', color='green', zorder=3,
            linewidth=linewidth, linestyle='--')
    ax.plot(x_test, mu_test_L, label='$\\hat\\mu_{Y_L}(x; \\theta_{\\mu_{Y_L}})$', color='blue', zorder=3,
            linewidth=linewidth, linestyle='--')
    if show_data:
        plt.scatter(x_hifi_data[:, 0], y_hifi_data, s=scatter_size, label='Hi-Fi Data', color='black', zorder=1,
                    linewidth=scatter_size)
    ax.set_xlabel(r'$x$', fontsize=axis_fontsize)
    ax.set_ylabel(r'$y$', fontsize=axis_fontsize)
    plt.legend(loc="lower left", fontsize=legend_fontsize)
    plt.grid()
    plt.tight_layout()
    if directory:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, "{:s}_plot.pdf".format(title)), dpi=1000)
        plt.savefig(os.path.join(directory, "{:s}_plot.tif".format(title)), dpi=1000)
        plt.close(fig)
    else:
        plt.show()

exp_num = "simple_scaled"
functional = 'polynomial'
poly_degree = 1
pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}".format(exp_num, functional, str(poly_degree)).replace('.', '_')

# --- Sweep every (mode_var_h, r) combination ---
results_rows = []

for mode_var_h_raw in mode_var_h_list:
    mode_var_h = mode_var_h_raw / (y_Global_Scaler.std ** 2)
    beta_0 = beta_calculator(alpha_0, mode_var_h)  # matches training script per mode_var_h

    for i, r in enumerate(r_store):
        n_hifi = n_H_store[i]
        directory = "{:s}/{:s}/degree_{:s}/mode_var_h_{:s}/r_{:s}".format(
            exp_num, functional, str(poly_degree), str(mode_var_h_raw), str(r)
        ).replace('.', '_')

        model = torch.load(directory + "/final", map_location=device)

        # Evaluate + plot for each region: entire, Hi-Fi data, no-Hi-Fi (prior)
        mse_entire, sig_entire, nll_entire = evaluate(model, start=0.0, end=1.0, steps=1000, scaled=scaled)
        mse_data, sig_data, nll_data = evaluate(model, start=0.0, end=0.5, steps=500, scaled=scaled)
        mse_prior, sig_prior, nll_prior = evaluate(model, start=0.5, end=1.0, steps=500, scaled=scaled)

        plot_results(model, "entire", start=0.0, end=1.0, steps=1000, scaled=scaled, show_data=True,
                     directory=directory)

        # Expected sigma^2 in the Hi-Fi region, given this (mode_var_h, r) combination
        exp_sig_data = (n_hifi * hifi_regions_noise_stds[0] ** 2 + 2 * beta_0) / (n_hifi + 2 * (alpha_0 + 1))
        # Expected sigma^2 in the no-Hi-Fi (prior) region is simply the prior mode itself
        exp_sig_prior = mode_var_h_raw

        results_rows.append({
            'mode_var_h': mode_var_h_raw,
            'r': r,
            'sig_data': sig_data,
            'exp_sig_data': exp_sig_data,
            'data_error': sig_data - exp_sig_data,
            'sig_prior': sig_prior,
            'exp_sig_prior': exp_sig_prior,
            'prior_error': sig_prior - exp_sig_prior,
            'mse_data': mse_data,
            'mse_prior': mse_prior,
            'mse_entire': mse_entire,
            'nll_data': nll_data,
            'nll_prior': nll_prior,
            'nll_entire': nll_entire,
        })

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(exp_num, 'variance_data_grid.csv'), index=False)

# --- Plot: Avg sigma^2 in the Hi-Fi region vs r, one line per mode_var_h_prior value ---
fig, ax = plt.subplots(dpi=1000)
for mode_var_h_raw in mode_var_h_list:
    subset = results_df[results_df['mode_var_h'] == mode_var_h_raw]
    ax.plot(subset['r'], subset['sig_data'], marker='x', linewidth=linewidth,
            label='$\\sigma^2_{{Y_H,\\text{{prior}}}} = {:.3g}$'.format(mode_var_h_raw))
ax.set_xlabel('$r$', fontsize=axis_fontsize)
ax.set_ylabel('Avg $\\hat\\sigma^2_{Y_H}(x; \\theta_{\\sigma^2_{Y_H}})$ in Hi-Fi Data Region', fontsize=axis_fontsize)
ax.set_xscale('log')
plt.legend(loc="upper right", fontsize=legend_fontsize)
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
ax.grid(True, which='both', axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(exp_num, "var_r_data_grid_plot.pdf"), dpi=1000, format='pdf')
plt.close(fig)

# --- Plot: Avg sigma^2 in the no-Hi-Fi (prior) region vs r, one line per mode_var_h_prior value ---
fig, ax = plt.subplots(dpi=1000)
for mode_var_h_raw in mode_var_h_list:
    subset = results_df[results_df['mode_var_h'] == mode_var_h_raw]
    ax.plot(subset['r'], subset['sig_prior'], marker='x', linewidth=linewidth,
            label='$\\sigma^2_{{Y_H,\\text{{prior}}}} = {:.3g}$'.format(mode_var_h_raw))
ax.set_xlabel('$r$', fontsize=axis_fontsize)
ax.set_ylabel('Avg $\\hat\\sigma^2_{Y_H}(x; \\theta_{\\sigma^2_{Y_H}})$ in No Hi-Fi Data Region', fontsize=axis_fontsize)
ax.set_xscale('log')
plt.legend(loc="upper right", fontsize=legend_fontsize)
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
ax.grid(True, which='both', axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(exp_num, "var_r_prior_grid_plot.pdf"), dpi=1000, format='pdf')
plt.close(fig)

# --- Heatmap: NLL over the entire domain, r vs mode_var_h_prior ---
pivot_nll = results_df.pivot(index='mode_var_h', columns='r', values='nll_entire')
fig, ax = plt.subplots(dpi=500)
im = ax.imshow(pivot_nll.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(pivot_nll.columns)))
ax.set_xticklabels([str(c) for c in pivot_nll.columns], rotation=90)
ax.set_yticks(range(len(pivot_nll.index)))
ax.set_yticklabels(['{:.3g}'.format(v) for v in pivot_nll.index])
ax.set_xlabel('$r$', fontsize=axis_fontsize)
ax.set_ylabel('$\\sigma^2_{Y_H,\\text{prior}}$', fontsize=axis_fontsize)
fig.colorbar(im, ax=ax, label='NLL (entire domain)')
plt.tight_layout()
plt.savefig(os.path.join(exp_num, "nll_heatmap_r_vs_prior.pdf"), dpi=1000, format='pdf')
plt.close(fig)