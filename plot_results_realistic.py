import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import unconcatenate_mu, unconcatenate_var, create_scalers, scale_data
import pickle
import mlflow

device = 'cpu'
torch.manual_seed(2025)
np.random.seed(2025)

# Set plotting parameters
title_fontsize = 10
axis_fontsize = 10
legend_fontsize = 8
other_linewidth = 1
linewidth = 1.25
scatter_size = 1.75

# Load the numpy data using pickle
with open('x_lofi_train_data.pkl', 'rb') as f:
    x_lofi_data = pickle.load(f)
with open('y_lofi_train_data.pkl', 'rb') as f:
    y_lofi_data = pickle.load(f)
with open('x_hifi_train_data.pkl', 'rb') as f:
    x_hifi_data = pickle.load(f)
with open('y_hifi_train_data.pkl', 'rb') as f:
    y_hifi_data = pickle.load(f)
with open('x_hifi_valid_data.pkl', 'rb') as f:
    x_valid_data = pickle.load(f)
with open('y_hifi_valid_data.pkl', 'rb') as f:
    y_valid_data = pickle.load(f)
with open('x_hifi_test_data.pkl', 'rb') as f:
    x_test_data = pickle.load(f)
with open('y_hifi_test_data.pkl', 'rb') as f:
    y_test_data = pickle.load(f)

# Squeeze relevant datasets
y_hifi_data = y_hifi_data.squeeze()
y_lofi_data = y_lofi_data.squeeze()
y_test_data = y_test_data.squeeze()
y_valid_data = y_valid_data.squeeze()

x_test_data = x_test_data.squeeze()
x_valid_data = x_valid_data.squeeze()

# Add labels to the x_lofi_data and x_hifi_data
x_lofi_data = torch.cat((x_lofi_data, torch.ones(x_lofi_data.shape[0], 1, dtype=torch.float32)),
                        dim=1)  # Add a column of ones for lofi label
x_hifi_data = torch.cat((x_hifi_data, torch.zeros(x_hifi_data.shape[0], 1, dtype=torch.float32)),
                        dim=1)  # Add a column of zeros for hifi label

# Create scalers
scalers = create_scalers(x_lofi_data, y_lofi_data)
x_Global_Scaler, y_Global_Scaler = scalers

# Scale data
x_hifi_scaled, y_hifi_scaled = scale_data(x_hifi_data, y_hifi_data, scalers)
x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
x_valid_scaled, y_valid_scaled = scale_data(x_valid_data.squeeze(), y_valid_data, scalers, label=False)

# Concatenate and store certain data sets together
x_H_train_data = torch.cat((x_hifi_scaled, x_lofi_scaled), 0)
y_H_train_data = torch.cat((y_hifi_scaled, y_lofi_scaled), 0)

# Create all training sets
H_train_data = [x_H_train_data.to(device), y_H_train_data.to(device)]
L_train_data = [x_lofi_scaled.to(device), y_lofi_scaled.to(device)]
D_train_data = [x_hifi_scaled.to(device), y_hifi_scaled.to(device)]

lofi_pretrain_data = [x_lofi_scaled.to(device), y_lofi_scaled.to(device)]
hifi_pretrain_data = [x_hifi_scaled.to(device), y_hifi_scaled.to(device)]
valid_data = [x_valid_scaled.to(device), y_valid_scaled.to(device)]

def plot_results(model, base_model, scalers):
    plot_model = model.to('cpu')
    plot_model.eval()
    base_model = base_model.to('cpu')
    base_model.eval()

    # Unpack scalers
    x_Global_Scaler = scalers[0]
    y_Global_Scaler = scalers[1]
    x_test_data_scaled = x_Global_Scaler.transform(x_test_data)

    # Plot Results on Test Data
    mu_test, var_test = plot_model(x_test_data_scaled.reshape(x_test_data_scaled.shape[-1], -1))
    mu_test = mu_test.detach()  # Detach and load on cpu
    sigma_test = var_test.detach()
    mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
    var_test_H, var_test_L = unconcatenate_var(sigma_test)

    # Base model
    mu_test_base, _ = base_model(x_test_data_scaled.reshape(x_test_data_scaled.shape[-1], -1))
    mu_test_base = mu_test_base.detach()  # Detach and load on cpu
    mu_test_base_H, _, _ = unconcatenate_mu(mu_test_base)

    if scalers:
        mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
        mu_test_L = y_Global_Scaler.inverse_transform(mu_test_L)
        mu_test_D = y_Global_Scaler.inverse_transform_mu_D(mu_test_D)
        var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)
        mu_test_base = y_Global_Scaler.inverse_transform(mu_test_base_H)

    # Sort data based on x_test_data
    sorted_indices = np.argsort(x_test_data)
    x_test_sorted = x_test_data[sorted_indices]
    y_test_sorted = y_test_data[sorted_indices]
    mu_test_D_sorted = mu_test_D.squeeze()[sorted_indices]
    mu_test_L_sorted = mu_test_L.squeeze()[sorted_indices]
    mu_test_H_sorted = mu_test_H.squeeze()[sorted_indices]
    var_test_L_sorted = var_test_L.squeeze()[sorted_indices]
    var_test_H_sorted = var_test_H.squeeze()[sorted_indices]
    mu_test_base_sorted = mu_test_base.squeeze()[sorted_indices]

    # Evaluate Sigma and MSE for entire test domain
    var_H_avg = np.mean(var_test_H.numpy())  # Evaluate and store mse and sigma avg for lofi model
    mse_H = torch.sum((mu_test_H.flatten() - y_test_data.flatten().squeeze().numpy()) ** 2) / len(mu_test.flatten())
    mse_base = torch.sum((mu_test_base.flatten() - y_test_data.flatten().squeeze().numpy()) ** 2) / len(mu_test_base.flatten())

    # Plot results
    fig, ax = plt.subplots(dpi=500)  # Create figure
    ax.fill_between(x_test_sorted, mu_test_H_sorted - var_test_H_sorted, mu_test_H_sorted + var_test_H_sorted, label = '$\mu_{Y_H}(w_f) \pm \sigma^2_{Y_H}(w_f)$', color='green', alpha=0.1, zorder=3)
    ax.plot(x_test_sorted, y_test_sorted, label='Ground Truth', color='orange', zorder=3,linewidth=other_linewidth)
    ax.plot(x_test_sorted, mu_test_base_sorted, label='$\mu_{Base}(w_f)$', color='blue', zorder=3, linewidth=other_linewidth)
    ax.plot(x_test_sorted, mu_test_H_sorted, label='$\mu_{Y_H}(w_f)$', color='green', zorder=3, linewidth=linewidth)
    #ax.plot(x_test_sorted, mu_test_D_sorted + mu_test_L_sorted, label='$\mu_{Y_L}(x) + \mu_{D}(x)$', color='red', zorder=3, linewidth=linewidth)
    plt.scatter(x_hifi_data[:,0], y_hifi_data, s=scatter_size, label='Hi-Fi Data', color='purple', zorder=1, linewidth=scatter_size)
    ax.set_xlabel(r'$w_f$', fontsize=axis_fontsize)
    ax.set_ylabel(r'$PR$', fontsize=axis_fontsize)
    plt.legend(loc="lower right", fontsize=legend_fontsize)
    plt.title(r'MSE $\mu_{{Y_H}}(w_f)$: {:.2g}    MSE $\mu_{{Base}}(w_f)$: {:.2g}'.format(mse_H, mse_base), fontsize=title_fontsize)
    plt.grid()
    #plt.tight_layout()
    plt.show()
    #plt.savefig("realistic_plot.svg", dpi=500)

model = torch.load(r"C:\Users\Suraj\Documents\01_git_repositories\01_phd_projects\EM_mismatch_approach\realistic_3_3_var_0_1_r_25\polynomial_polynomial_on_mu_l\degree_3_mu_l_degree_3\r_25_0\final")
base_model = torch.load(r"C:\Users\Suraj\Documents\01_git_repositories\01_phd_projects\EM_mismatch_approach\realistic_3_3_var_0_1_r_25\polynomial_polynomial_on_mu_l\degree_3_mu_l_degree_3\mse_model")
plot_results(model, base_model, scalers)