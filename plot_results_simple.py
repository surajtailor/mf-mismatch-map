import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import Scaler, beta_calculator, desired_ratio_var_H_overlap, desired_var_0_mu_H_overlap, unconcatenate_mu, unconcatenate_var, create_scalers, scale_data
from data_generation import generate_pretrain_lofi_data, generate_lofi_data, generate_hifi_lofi_data, generate_test_data
import pandas as pd
import matplotlib.ticker as ticker

scaled = True

device = 'cpu'
torch.manual_seed(2025)
np.random.seed(2025)

# Set plotting parameters
title_fontsize = 10
axis_fontsize = 10
legend_fontsize = 8
other_linewidth = 1
linewidth = 1.25
scatter_size = 0.75

def function(x):
    # np array of zeros
    return torch.sin(3 * np.pi * x) + 0.5 * x + 1 / 4
def lofi_function(x):
    return torch.sin(3 * np.pi * x)

## Data Generation
pre_train_lofi_regions = [(0, 1)]  # Domain of interest for plotting test
pre_train_lofi_num_points = [1000]
pre_train_lofi_total_num_points = sum(pre_train_lofi_num_points)

# Set regions where its only Lofi data
lofi_regions = [(0.5, 1)]
lofi_regions_num_points = [500]
lofi_total_num_points = sum(lofi_regions_num_points)

hifi_regions = [(0, 0.5)]
hifi_regions_num_points = [500]
hifi_total_num_points = sum(hifi_regions_num_points)
hifi_regions_noise_stds = [0.05]  # Small amounts of noise
hifi_regions_noise_mu = [0]

valid_regions = (0, 1)
valid_regions_num_points = 100

test_regions = (0, 1)
test_regions_num_points = 1000

# Create data
x_lofi_pretrain_data, y_lofi_pretrain_data = generate_pretrain_lofi_data(pre_train_lofi_total_num_points, pre_train_lofi_regions,pre_train_lofi_num_points, lofi_function)
x_lofi_only_data, y_lofi_only_data = generate_lofi_data(lofi_total_num_points, lofi_regions, lofi_regions_num_points,lofi_function)
x_hifi_data, y_hifi_data, x_lofi_match_data, y_lofi_match_data = generate_hifi_lofi_data(hifi_total_num_points,hifi_regions,hifi_regions_num_points,hifi_regions_noise_mu, hifi_regions_noise_stds, function, lofi_function)
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
n_lofi = 1  # NEVER CHANGED FROM
mode_var_h = 2  # Value of uncertainty that it defaults to, after calculating the prior. #Need
alpha_0 = 1  # Can set this to whatever we want. Relevant when we have more fidelities in the data, from which we can weight things relative to each other.
beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.
var_H = 0.2  # This is the max value for the Variance on the H value. Not 'mismatch + H uncertainty' value should be less than mode_var_h.

r_store = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]
var_0_store = []
n_H_store = []
for r in r_store:
    r_mu = r  # Data is 20 times more important than the prior
    r_var = r_mu  # Data is at least 20 times more important than the prior.
    # Calculate values for n_H for desired ratio for r_mu
    n_H_var = desired_ratio_var_H_overlap(r_var, alpha_0)
    var_0 = desired_var_0_mu_H_overlap(r_mu, n_H_var, var_H)  # Since n_values have to be the same.
    var_0_store.append(var_0)
    n_H_store.append(n_H_var)

def plot_results(model, title, start = 0, end = 1, steps = 1000, scaled = False, show_data = False, directory = None):
    plot_model = model.to('cpu')
    plot_model.eval()

    if scaled == False:
        x_test = torch.linspace(start, end, steps)  # Values for test x domain test
        y_test = torch.tensor(function(x_test))  # Value for y data in test

        mu_test, var_test = plot_model(x_test.reshape(x_test.shape[-1], -1))
        mu_test = mu_test.detach()  # Detach and load on cpu
        sigma_test = var_test.detach()
        mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
        var_test_H, var_test_L = unconcatenate_var(sigma_test)

    if scaled == True:
        x_Global_Scaler = scalers[0]
        y_Global_Scaler = scalers[1]
        x_test = torch.linspace(start, end, steps)  # Values for test x domain test
        y_test = torch.tensor(function(x_test))  # Value for y data in test
        x_test_input = x_Global_Scaler.transform(x_test)

        # Plot Results on Test Data
        mu_test, var_test = plot_model(x_test_input.reshape(x_test_input.shape[-1], -1))
        mu_test = mu_test.detach()  # Detach and load on cpu
        sigma_test = var_test.detach()
        mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
        var_test_H, var_test_L = unconcatenate_var(sigma_test)

        mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
        mu_test_L = y_Global_Scaler.inverse_transform(mu_test_L)
        mu_test_D = y_Global_Scaler.inverse_transform_mu_D(mu_test_D)
        var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)

    sig_test_H = np.sqrt(var_test_H.numpy())  # Convert to numpy for plotting

    # Evaluate Sigma and MSE for entire test domain
    var_H_avg = np.mean(var_test_H.numpy())  # Evaluate and store mse and sigma avg for lofi model
    mse_H = torch.sum((mu_test_H.flatten() - y_test.flatten().squeeze().numpy()) ** 2) / len(mu_test_H.flatten())

    # Plot results
    fig, ax = plt.subplots(dpi=500)  # Create figure
    ax.fill_between(x_test, mu_test_H - 2*sig_test_H, mu_test_H + 2*sig_test_H ,label='$\mu_{Y_H}(x) \pm 2\sigma_{Y_H}(x)$', color='green', alpha=0.1, zorder=3)
    ax.plot(x_test, mu_test_L + mu_test_D, label='$\mu_{Y_L}(x)$ + $\mu_{D}(x)$', color='red', alpha=0.5, zorder=3, linewidth=other_linewidth)
    ax.plot(x_test, y_test, label='Ground Truth', color='orange', zorder=3, linewidth=other_linewidth)
    ax.plot(x_test, mu_test_H, label='$\mu_{Y_H}(x)$', color='green', zorder=3, linewidth=linewidth)
    #add mu l plot
    ax.plot(x_test, mu_test_L, label='$\mu_{Y_L}(x)$', color='blue', zorder=3, linewidth=linewidth)
    if show_data == True:
        plt.scatter(x_hifi_data[:,0], y_hifi_data, s=scatter_size, label='Hi-Fi Data', color='purple', zorder=1, linewidth=scatter_size)
    ax.set_xlabel(r'$x$', fontsize=axis_fontsize)
    ax.set_ylabel(r'$y$', fontsize=axis_fontsize)
    plt.legend(loc="lower right", fontsize=legend_fontsize)
    plt.title("$r$ = {:s}".format(title), fontsize=title_fontsize)
    plt.grid()
    if directory:
        plt.savefig(directory + r"\{:s}_plot.png".format(title), dpi=500)
    else:
        plt.show()
    return mse_H, var_H_avg

list = ['r_0_1', 'r_0_2', 'r_0_3', 'r_0_5', 'r_1_0', 'r_2_0', 'r_3_0', 'r_5_0', 'r_10_0', 'r_20_0', 'r_30_0', 'r_50_0', 'r_100_0']
label_list = ['0.1', '0.2', '0.3', '0.5', '1.0', '2.0', '3.0', '5.0', '10.0', '20.0', '30.0', '50.0', '100.0']

# Empty storage
mse_all_store = []
sig_all_store = []
mse_data_store = []
sig_data_store = []
mse_prior_store = []
sig_prior_store = []

## Plot results for each r value and save them
for i in range(len(list)):
    r_x = list[i]
    title = label_list[i]
    model = torch.load(r"\polynomial\degree_1\{:s}\final".format(r_x))

    # Evaluate and plot models
    mse_entire, var_avg_entire = plot_results(model, title, start = 0.0, end = 1.0, steps = 1000, scaled = scaled, show_data = True, directory = r"\polynomial\degree_1\{:s}".format(r_x))
    mse_data, var_avg_data = plot_results(model, "Data", start=0.0, end=0.5, steps = 500, scaled = scaled)
    mse_prior, var_avg_prior = plot_results(model, "Prior", start=0.5, end=1.0, steps = 500, scaled = scaled)

    # Store results
    mse_all_store.append(mse_entire)
    sig_all_store.append(var_avg_entire)
    mse_data_store.append(mse_data)
    sig_data_store.append(var_avg_data)
    mse_prior_store.append(mse_prior)
    sig_prior_store.append(var_avg_prior)

## Save expected variance values for data regions, using the formula derived from the Inverse Gamma distribution, for each r value. This is the expected variance value for the data regions, given the prior and the amount of data in those regions.
exp_var_values_data_regions = []
for r in r_store:
    r_mu = r  # Data is 20 times more important than the prior
    r_var = r_mu  # Data is at least 20 times more important than the prior.
    n_H_var = desired_ratio_var_H_overlap(r_var, alpha_0)
    exp_var_data_regions = (n_H_var*hifi_regions_noise_stds[0]**2 + 2*beta_0)/(n_H_var + 2*(alpha_0+1))
    exp_var_values_data_regions.append(exp_var_data_regions)

# Save data to a CSV file.
data_error_store = np.array(sig_data_store) - np.array(exp_var_values_data_regions)
prior_error_store = np.array(sig_prior_store) - 2

# Create a dictionary to store the data
data = {
    'r': r_store,
    'sig_data_store': sig_data_store,
    'exp_sig_store': exp_var_values_data_regions,
    'data_error_store': data_error_store,
    'sig_prior_store': sig_prior_store,
    'prior_error_store': prior_error_store,
    'sig_all_store': sig_all_store,
    'mu_data_store': mse_data_store,
    'mu_prior_store': mse_prior_store,
    'mu_all_store': mse_all_store
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(r'variance_data.csv', index=False)

## Plot Sigma and MSE for all three models
fig, ax = plt.subplots(dpi=500)  # Create figure
ax.plot(r_store, sig_data_store, label='Avg $\sigma^2_{Y_H}(x)$ in Hi-Fi Data Region', color='green', marker='x', linewidth = linewidth)
ax.plot(r_store, sig_prior_store, label='Avg $\sigma^2_{Y_H}(x)$ in No Hi-Fi Data Region', color='blue', marker='x', linewidth = linewidth)
#ax.plot(r_store, exp_var_values_data_regions, label=r'Exp $\sigma^2_{Y_H}(x)$ in Hi-Fi Data Region$', linewidth = linewidth, color='orange', marker='x', alpha = 0.3)

ax.set_xlabel('$r$', fontsize=axis_fontsize)
ax.set_ylabel('Avg $\sigma^2_{Y_H}(x)$', fontsize=axis_fontsize)
plt.legend(loc="center right", fontsize=legend_fontsize)
plt.title("Avg $\sigma^2_{Y_H}(x)$ against choice of $r$", fontsize=title_fontsize)

# # Increase number of y-axis ticks (notches)
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))  # Adjust nbins for more/less ticks
# Enable gridlines at y-axis ticks
ax.grid(True, which='both', axis='y', linestyle='--')
plt.savefig(r"var_r_plot")