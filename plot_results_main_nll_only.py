import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import unconcatenate_mu, unconcatenate_var, create_scalers, scale_data, beta_calculator, desired_ratio_var_H_overlap, desired_var_0_mu_H_overlap
from data_generation import generate_pretrain_lofi_data, generate_lofi_data, generate_hifi_lofi_data, generate_test_data
import pandas as pd
import mlflow
import os

device = 'cpu'
torch.manual_seed(2025)
np.random.seed(2025)

# Set plotting parameters
title_fontsize = 12
axis_fontsize = 14
legend_fontsize = 9
other_linewidth = 1
linewidth = 1.25
scatter_size = 0.75

for exp_num in ["exp_1", "exp_2", "exp_3", "exp_4"]:
    if exp_num == "exp_1":
        def function(x):
            return (x - np.sqrt(2)) * (np.sin(8 * np.pi * x)) ** 2
        def lofi_function(x):
            return (x - np.sqrt(2)) * (np.sin(8 * np.pi * x)) ** 2 + x - 2

    if exp_num == "exp_2":
        def function(x):
            return (x - np.sqrt(2)) * (np.sin(8 * np.pi * x)) ** 2
        def lofi_function(x):
            return (np.sin(8 * np.pi * x))

    if exp_num == "exp_3":
        def function(x):
            return x ** 2 + (np.sin(8 * np.pi * x + np.pi / 10)) ** 2
        def lofi_function(x):
            return (np.sin(8 * np.pi * x))

    if exp_num == "exp_4":
        def function(x):
            return 0.5 * ((6 * x - 2) ** 2) * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5
        def lofi_function(x):
            return ((6 * x - 2) ** 2) * np.sin(12 * x - 4)

    ## Data Generation
    pre_train_lofi_regions = [(0, 1)]  # Domain of interest for plotting test
    pre_train_lofi_num_points = [1000]
    pre_train_lofi_total_num_points = sum(pre_train_lofi_num_points)

    # Set regions where its only Lofi data
    lofi_regions = [(.4, .6), (.8, 1)]
    lofi_regions_num_points = [100, 100]
    lofi_total_num_points = sum(lofi_regions_num_points)

    hifi_regions = [(0, .4), (.6, .8)]
    hifi_regions_num_points = [200, 100]
    hifi_total_num_points = sum(hifi_regions_num_points)
    hifi_regions_noise_stds = [0.01, 0.01]  # Small amounts of noise
    hifi_regions_noise_mu = [0, 0]

    # Create data
    x_lofi_pretrain_data, y_lofi_pretrain_data = generate_pretrain_lofi_data(pre_train_lofi_total_num_points, pre_train_lofi_regions, pre_train_lofi_num_points, lofi_function)
    x_lofi_only_data, y_lofi_only_data = generate_lofi_data(lofi_total_num_points, lofi_regions, lofi_regions_num_points, lofi_function)
    x_hifi_data, y_hifi_data, x_lofi_match_data, y_lofi_match_data = generate_hifi_lofi_data(hifi_total_num_points, hifi_regions, hifi_regions_num_points, hifi_regions_noise_mu, hifi_regions_noise_stds, function, lofi_function)

    # Concatenate lofi match x and y data before placing into []
    x_lofi_data = torch.cat((x_lofi_match_data, x_lofi_only_data), 0)
    y_lofi_data = torch.cat((y_lofi_match_data, y_lofi_only_data), 0)

    # Create scalers
    scalers = create_scalers(x_lofi_pretrain_data, y_lofi_pretrain_data)
    x_Global_Scaler, y_Global_Scaler = scalers

    # --- NLL evaluation functions (mirror mse_data / mse_custom) ---
    def gaussian_nll(input, target, var, eps=1e-6, reduction='mean'):
        var = var.clamp(min=eps)
        loss = 0.5 * (torch.log(var) + (input - target).pow(2) / var)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    def nll_data(model, scalers):
        """NLL over the Hi-Fi data region: x in [0,0.4] U [0.6,0.8]."""
        plot_model = model.to('cpu')
        plot_model.eval()

        x_test_1 = torch.linspace(0, 0.4, 500)
        x_test_2 = torch.linspace(0.6, 0.8, 500)
        x_test = torch.cat((x_test_1, x_test_2), 0)
        y_test = torch.tensor(function(x_test))

        x_Global_Scaler = scalers[0]
        y_Global_Scaler = scalers[1]
        x_test_scaled = x_Global_Scaler.transform(x_test)

        mu_test, var_test = plot_model(x_test_scaled.reshape(x_test_scaled.shape[-1], -1))
        mu_test = mu_test.detach()
        sigma_test = var_test.detach()
        mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
        var_test_H, var_test_L = unconcatenate_var(sigma_test)

        if scalers:
            mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
            var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)

        return gaussian_nll(mu_test_H, y_test, var_test_H)

    def nll_prior(model, scalers):
        """NLL over the non-Hi-Fi (prior) region: x in [0.4,0.6] U [0.8,1]."""
        plot_model = model.to('cpu')
        plot_model.eval()

        x_test_1 = torch.linspace(0.4, 0.6, 500)
        x_test_2 = torch.linspace(0.8, 1.0, 500)
        x_test = torch.cat((x_test_1, x_test_2), 0)
        y_test = torch.tensor(function(x_test))

        x_Global_Scaler = scalers[0]
        y_Global_Scaler = scalers[1]
        x_test_scaled = x_Global_Scaler.transform(x_test)

        mu_test, var_test = plot_model(x_test_scaled.reshape(x_test_scaled.shape[-1], -1))
        mu_test = mu_test.detach()
        sigma_test = var_test.detach()
        mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
        var_test_H, var_test_L = unconcatenate_var(sigma_test)

        if scalers:
            mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
            var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)

        return gaussian_nll(mu_test_H, y_test, var_test_H)

    def nll_entire(model, scalers, start=0, end=1, steps=1000):
        """NLL over the entire domain."""
        plot_model = model.to('cpu')
        plot_model.eval()

        x_test = torch.linspace(start, end, steps)
        y_test = torch.tensor(function(x_test))

        x_Global_Scaler = scalers[0]
        y_Global_Scaler = scalers[1]
        x_test_scaled = x_Global_Scaler.transform(x_test)

        mu_test, var_test = plot_model(x_test_scaled.reshape(x_test_scaled.shape[-1], -1))
        mu_test = mu_test.detach()
        sigma_test = var_test.detach()
        mu_test_H, mu_test_L, mu_test_D = unconcatenate_mu(mu_test)
        var_test_H, var_test_L = unconcatenate_var(sigma_test)

        if scalers:
            mu_test_H = y_Global_Scaler.inverse_transform(mu_test_H)
            var_test_H = y_Global_Scaler.inverse_transform_variance(var_test_H)

        return gaussian_nll(mu_test_H, y_test, var_test_H)
    # --- end NLL functions ---

    # Prior Hyper Parameter = 0  # LEGACY NEVER CHANGED FROM 0
    n_lofi = 1  # NEVER CHANGED FROM 1
    mode_var_h = 1  # Value of uncertainty that it defaults to, after calculating the prior. #Nee
    alpha_0 = 1  # Can set this to whatever we want. Relevant when we have more fidelities in the data, from which we can weight things relative to each other.
    beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.
    var_H = 0.05  # This is the max value for the Variance on the H value. Not 'mismatch + H uncertainty' value should be less than mode_var_h.

    # Scale the mode_var_h, beta_0, and var_H
    mode_var_h = mode_var_h / (y_Global_Scaler.std ** 2)  # Value of uncertainty that it defaults to, after calculating the prior. #Nee
    beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.
    var_H = var_H / (y_Global_Scaler.std ** 2)  # This is the max value for the Variance on the H value.

    r_store = [50]
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

    # Set List of Functionals
    polynomial_degree_list = [1, 3, 5, 10]
    polynomial_on_mu_l_degree_list = [1, 3, 5, 10]
    polynomial_degree_list_overlap = [1, 3, 5]
    polynomial_on_mu_l_degree_list_overlap = [1, 3, 5]
    fourier_degree_list = [5]

    mse_data_regions = []
    mse_inter_regions = []
    mse_extra_regions = []
    mse_entire_regions = []

    # NLL stores
    nll_data_regions = []
    nll_prior_regions = []
    nll_entire_regions = []

    # #Polynomial Functional
    for poly_degree in polynomial_degree_list:
        functional = 'polynomial'
        pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}".format(exp_num, functional, str(poly_degree)).replace('.', '_')
        base_model = torch.load(pre_train_lofi_directory + "/mse_model")

        for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
            directory = "{:s}/{:s}/degree_{:s}/r_{:s}".format(exp_num, functional, str(poly_degree), str(n_hifi / (2*(alpha_0+1)))).replace('.', '_')
            model = torch.load(directory + "/final")

            nll_d = nll_data(model, scalers)
            nll_p = nll_prior(model, scalers)
            nll_e = nll_entire(model, scalers, start=0, end=1, steps=1000)

            nll_data_regions.append(nll_d)
            nll_prior_regions.append(nll_p)
            nll_entire_regions.append(nll_e)

    # Polynomial on mu_l Functional
    for poly_on_mu_l_degree in polynomial_on_mu_l_degree_list:
        functional = 'polynomial_on_mu_l'
        pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}".format(exp_num, functional, str(poly_on_mu_l_degree)).replace('.', '_')
        base_model = torch.load(pre_train_lofi_directory + "/mse_model")

        for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
            directory = "{:s}/{:s}/degree_{:s}/r_{:s}".format(exp_num, functional, str(poly_on_mu_l_degree), str(n_hifi / (2*(alpha_0+1)))).replace('.', '_')
            model = torch.load(directory + "/final")

            nll_d = nll_data(model, scalers)
            nll_p = nll_prior(model, scalers)
            nll_e = nll_entire(model, scalers, start=0, end=1, steps=1000)

            nll_data_regions.append(nll_d)
            nll_prior_regions.append(nll_p)
            nll_entire_regions.append(nll_e)

    # # Fourier Functional
    for fourier_degree in fourier_degree_list:
        functional = 'periodic_fourier'
        pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}".format(exp_num, functional, str(fourier_degree)).replace('.', '_')
        base_model = torch.load(pre_train_lofi_directory + "/mse_model")

        for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
            n_count = [n_hifi, n_lofi]  # Hifi and Lofi Prior
            # Create results directory
            directory = "{:s}/{:s}/degree_{:s}/r_{:s}".format(exp_num,functional, str(fourier_degree), str(n_hifi/(2*(alpha_0+1)))).replace('.', '_')
            model = torch.load(directory + "/final")

            nll_d = nll_data(model, scalers)
            nll_p = nll_prior(model, scalers)
            nll_e = nll_entire(model, scalers, start=0, end=1, steps=1000)

            nll_data_regions.append(nll_d)
            nll_prior_regions.append(nll_p)
            nll_entire_regions.append(nll_e)

    # Periodic Functional
    functional = 'periodic'
    pre_train_lofi_directory = "{:s}/{:s}".format(exp_num, functional).replace('.', '_')
    torch.load(pre_train_lofi_directory + "/mse_model")

    for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
        n_count = [n_hifi, n_lofi]  # Hifi and Lofi Prior
        # Create results directory
        directory = "{:s}/{:s}/r_{:s}".format(exp_num, functional, str(n_hifi/(2*(alpha_0+1)))).replace('.', '_')
        model = torch.load(directory + "/final")

        nll_d = nll_data(model, scalers)
        nll_p = nll_prior(model, scalers)
        nll_e = nll_entire(model, scalers, start=0, end=1, steps=1000)

        nll_data_regions.append(nll_d)
        nll_prior_regions.append(nll_p)
        nll_entire_regions.append(nll_e)

    #Poly and Poly on Mu L Combined Functionals
    for poly_degree in polynomial_degree_list_overlap:
        for poly_on_mu_l_degree in polynomial_on_mu_l_degree_list_overlap:
            functional = 'polynomial_polynomial_on_mu_l'
            pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}_mu_l_degree_{:s}".format(exp_num, functional,str(poly_degree),str(poly_on_mu_l_degree)).replace('.', '_')
            base_model = torch.load(pre_train_lofi_directory + "/mse_model")

            for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
                n_count = [n_hifi, n_lofi]  # Hifi and Lofi Prior
                directory = "{:s}/{:s}/degree_{:s}_mu_l_degree_{:s}/r_{:s}".format(exp_num, functional, str(poly_degree), str(poly_on_mu_l_degree), str(n_hifi/(2*(alpha_0+1)))).replace('.', '_')
                model = torch.load(directory + "/final")

                nll_d = nll_data(model, scalers)
                nll_p = nll_prior(model, scalers)
                nll_e = nll_entire(model, scalers, start=0, end=1, steps=1000)

                nll_data_regions.append(nll_d)
                nll_prior_regions.append(nll_p)
                nll_entire_regions.append(nll_e)

    # #Poly and Periodic Combined Functionals
    for poly_degree in polynomial_degree_list_overlap:
        functional = 'polynomial_periodic'
        pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}".format(exp_num, functional, str(poly_degree)).replace('.', '_')
        base_model = torch.load(pre_train_lofi_directory + "/mse_model")

        for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
            n_count = [n_hifi, n_lofi]  # Hifi and Lofi Prior
            # Create results directory
            directory = "{:s}/{:s}/degree_{:s}/r_{:s}".format(exp_num, functional, str(poly_degree), str(n_hifi/(2*(alpha_0+1)))).replace('.', '_')
            model = torch.load(directory + "/final")

            nll_d = nll_data(model, scalers)
            nll_p = nll_prior(model, scalers)
            nll_e = nll_entire(model, scalers, start=0, end=1, steps=1000)

            nll_data_regions.append(nll_d)
            nll_prior_regions.append(nll_p)
            nll_entire_regions.append(nll_e)

    results_dict = {
        'nll_data_regions': nll_data_regions,
        'nll_prior_regions': nll_prior_regions,
        'nll_entire_regions': nll_entire_regions,
    }

    results_dict = {key: [value.item() for value in values] for key, values in results_dict.items()}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('{:s}/nll_data.csv'.format(exp_num))