import torch
import numpy as np
import matplotlib.pyplot as plt
from training_functions import trainer, pretrainer
from net import NetAnyFunctional
from utilities import Scaler, beta_calculator, desired_ratio_var_H_overlap, desired_var_0_mu_H_overlap, create_scalers, scale_data
import os
import mlflow
import pickle

device = 'cpu'
torch.manual_seed(2025)
np.random.seed(2025)

# Turn on MLflow (if necessary)
# mlflow_process = subprocess.Popen(["mlflow", "ui"])  # Start MLflow UI in the background
# mlflow.set_tracking_uri(r"C:\Users\Suraj\Documents\01_git_repositories\01_phd_projects\toy_mismatch_two_outputs_functional")
# nohup mlflow ui --port 65098 --host 127.0.0.1 >mlflow.log 2>&1 &

for exp_num in ["pressure_ratio"]:
    os.makedirs(exp_num, exist_ok=True)

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
    x_lofi_data = torch.cat((x_lofi_data, torch.ones(x_lofi_data.shape[0], 1, dtype=torch.float32)), dim=1)  # Add a column of ones for lofi label
    x_hifi_data = torch.cat((x_hifi_data, torch.zeros(x_hifi_data.shape[0], 1, dtype=torch.float32)), dim=1)  # Add a column of zeros for hifi label

    # Create scalers
    scalers = create_scalers(x_lofi_data, y_lofi_data)
    x_Global_Scaler, y_Global_Scaler = scalers

    # Scale data
    x_hifi_scaled, y_hifi_scaled = scale_data(x_hifi_data, y_hifi_data, scalers)
    x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
    x_valid_scaled, y_valid_scaled = scale_data(x_valid_data.squeeze(), y_valid_data, scalers, label = False)

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

    # Prior Hyper Parameter = 0  # LEGACY NEVER CHANGED FROM 0
    # Prior Hyper Parameter = 0  # LEGACY NEVER CHANGED FROM 0
    n_lofi = 1  # NEVER CHANGED FROM 1
    mode_var_h = 0.1  # Value of uncertainty that it defaults to, after calculating the prior. #Need
    alpha_0 = 1  # Can set this to whatever we want. Relevant when we have more fidelities in the data, from which we can weight things relative to each other.
    beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.
    var_H = 0.05  #

    # Scale the mode_var_h, beta_0, and var_H
    mode_var_h = mode_var_h / (y_Global_Scaler.std ** 2)  # Value of uncertainty that it defaults to, after calculating the prior. #Nee
    beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.
    var_H = var_H / (y_Global_Scaler.std ** 2)  # This is the max value for the Variance on the H value.

    r_store = [10]
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

    # Training Parameters
    pre_train_iter = 5000
    train_n_iter = 10000
    lr = 1e-3
    num_workers = 0
    batch_size = 40

    # Num layers and hidden units
    num_layers = 6
    hidden_size = 20
    activation_func = 'elu'

    # Plot frequency
    valid_freq = 1000

    # Set List of Functionals
    polynomial_degree_list_overlap = [3]
    polynomial_on_mu_l_degree_list_overlap = [3]

    # Not used anywhere, just listed.
    individuals_functionals = ['polynomial', 'polynomial_on_mu_l', 'periodic', 'periodic_fourier']
    combined_functionals = ['polynomial_periodic', 'polynomial_polynomial_on_mu_l', 'polynomial_on_mu_l_periodic', 'polynomial_polynomial_on_mu_l_periodic']

    # #Poly and Poly on Mu L Combined Functionals
    for poly_degree in polynomial_degree_list_overlap:
        for poly_on_mu_l_degree in polynomial_on_mu_l_degree_list_overlap:
            functional = 'polynomial_polynomial_on_mu_l'
            model = NetAnyFunctional(input_size=1, num_layers=num_layers, hidden_size=hidden_size, device=device,functional=functional, poly_degree=poly_degree, fourier_degree=None, poly_mu_l_degree=poly_on_mu_l_degree, activation_func=activation_func).to(device)
            pre_train_lofi_directory = "{:s}/{:s}/degree_{:s}_mu_l_degree_{:s}".format(exp_num, functional,str(poly_degree),str(poly_on_mu_l_degree)).replace('.', '_')
            os.makedirs(pre_train_lofi_directory, exist_ok=True)
            run_name = "1"
            experiment_name = pre_train_lofi_directory

            # #Pretrain model
            pretrainer(model, hifi_pretrain_data, lofi_pretrain_data, valid_data, "pretrain", "model", device=device,n_iter=pre_train_iter, learning_rate=lr, scheduler_type="StepLR", save_freq=1000, valid_freq=valid_freq, num_workers=num_workers, batch_size=batch_size)
            torch.save(model, pre_train_lofi_directory + "/mse_model")

            for i, (n_hifi, var_0) in enumerate(zip(n_H_store, var_0_store)):
                model = torch.load(pre_train_lofi_directory + "/mse_model", map_location=device)
                n_count = [n_hifi, n_lofi]  # Hifi and Lofi Prior
                directory = "{:s}/{:s}/degree_{:s}_mu_l_degree_{:s}/r_{:s}".format(exp_num, functional, str(poly_degree), str(poly_on_mu_l_degree), str(n_hifi/(4*(alpha_0+1)))).replace('.', '_')
                os.makedirs(directory, exist_ok=True)
                run_name = "1"
                experiment_name = directory

                # Normal trainer
                run_id = trainer(model, H_train_data, L_train_data, D_train_data, valid_data, experiment_name, "model",
                                 hyperparameters=[var_0, alpha_0,
                                                  beta_0, n_count], run_name=run_name, device=device, n_iter=train_n_iter,
                                 learning_rate=lr, scheduler_type="StepLR", save_freq=1000,
                                 valid_freq=valid_freq, num_workers=num_workers, batch_size=batch_size)
                torch.save(model, directory + "/final")