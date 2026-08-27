import torch
import numpy as np
import matplotlib.pyplot as plt
from training_functions import trainer, pretrainer
from net import NetAnyFunctional
from utilities import Scaler, beta_calculator, desired_ratio_var_H_overlap, desired_var_0_mu_H_overlap, create_scalers, scale_data
from data_generation import generate_hifi_lofi_data, generate_lofi_data, generate_pretrain_lofi_data, generate_test_data
from torch.utils.data import DataLoader
import os
import mlflow

device = 'cpu'
torch.manual_seed(2025)
np.random.seed(2025)

# Turn on MLflow (if necessary)
# mlflow_process = subprocess.Popen(["mlflow", "ui"])  # Start MLflow UI in the background
# mlflow.set_tracking_uri(r"C:\Users\Suraj\Documents\01_git_repositories\01_phd_projects\toy_mismatch_two_outputs_functional")
# nohup mlflow ui --port 65098 --host 127.0.0.1 >mlflow.log 2>&1 &

for exp_num in ["simple_scaled"]:
    os.makedirs(exp_num, exist_ok=True)
    if exp_num == "simple_scaled":
        def function(x):
            # np array of zeros
            return torch.sin(3 * np.pi * x) + 0.5 *x + 1/4
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
    x_valid_scaled, y_valid_scaled = scale_data(x_valid_data, y_valid_data, scalers, label = False)

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
    alpha_0 = 1  # Can set this to whatever we want. Relevant when we have more fidelities in the data, from which we can weight things relative to each other.
    var_H = 0.000001  # This is the max value for the Variance on the H value. Not 'mismatch + H uncertainty' value should be less than mode_var_h.

    # Set List of Functionals
    polynomial_degree_list = [1]

    #loop through values of mode_var_H
    mode_var_h_list = [0.0002, 0.0001, 0.00005, 0.00002, 0.00001] # Value of uncertainty that it defaults to, after calculating the prior. #Need
    r_store = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]
    mode_var_h_store = []
    beta_0_store = []
    var_H_store = []
    var_0_store = []
    n_H_store = []

    # Training Parameters
    pre_train_iter = 2000
    train_n_iter = 1000
    lr = 1e-3
    num_workers = 0
    batch_size = 80

    # Num layers and hidden units
    num_layers = 6
    hidden_size = 20
    activation_func = 'elu'

    # Tau adjustment
    tau_adjustment = True
    tau = 0.8

    # Plot frequency
    valid_freq = 1000

    var_H = 0.01  # max value for Variance on H
    var_H_scaled = var_H / (y_Global_Scaler.std ** 2)  # scaled ONCE, not inside the mode_var_h loop

    for poly_degree in polynomial_degree_list:
        functional = 'polynomial'
        model = NetAnyFunctional(input_size=1, num_layers=num_layers, hidden_size=hidden_size, device=device,
                                 functional=functional, poly_degree=poly_degree, fourier_degree=None,
                                 poly_mu_l_degree=None, activation_func=activation_func).to(device)
        pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}".format(exp_num, functional, str(poly_degree)).replace('.',
                                                                                                                 '_')
        os.makedirs(pre_train_lofi_directory, exist_ok=True)

        # Pretrain once per poly_degree (unaffected by mode_var_h / r)
        pretrainer(model, hifi_pretrain_data, lofi_pretrain_data, valid_data, "pretrain", "model", device=device,
                   n_iter=pre_train_iter, learning_rate=lr, scheduler_type="StepLR",
                   save_freq=1000, valid_freq=valid_freq, num_workers=num_workers, batch_size=batch_size)
        torch.save(model, pre_train_lofi_directory + "/mse_model")

        for mode_var_h_raw in mode_var_h_list:
            mode_var_h = mode_var_h_raw / (y_Global_Scaler.std ** 2)
            beta_0 = beta_calculator(alpha_0, mode_var_h)

            for r in r_store:
                r_mu = r
                r_var = r_mu
                n_H_var = desired_ratio_var_H_overlap(r_var, alpha_0)
                var_0 = desired_var_0_mu_H_overlap(r_mu, n_H_var, var_H_scaled)
                n_count = [n_H_var, n_lofi]

                model = torch.load(pre_train_lofi_directory + "/mse_model", map_location=device)

                directory = "{:s}/{:s}/degree_{:s}/mode_var_h_{:s}/r_{:s}".format(exp_num, functional, str(poly_degree), str(mode_var_h_raw), str(r)).replace('.', '_')
                os.makedirs(directory, exist_ok=True)
                run_name = "1"
                experiment_name = directory

                run_id = trainer(model, H_train_data, L_train_data, D_train_data, valid_data, experiment_name, "model",
                                 hyperparameters=[var_0, alpha_0, beta_0, n_count], run_name=run_name, device=device,
                                 n_iter=train_n_iter, learning_rate=lr, scheduler_type="StepLR", save_freq=1000,
                                 valid_freq=valid_freq, num_workers=num_workers, batch_size=batch_size,
                                 tau_adjustment=tau_adjustment, tau=tau)
                torch.save(model, directory + "/final")