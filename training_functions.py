import torch
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR
from loss_functions import loss_mse, loss_fn ,unconcatenate_var, unconcatenate_mu
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

def scheduler_case(scheduler_type, optimizer, initial_lr, num_epochs, final_lr = 1e-4, steps = 3):

    if scheduler_type == "ExponentialLR":
        gamma_value = (final_lr/initial_lr)**(1/num_epochs)
        return ExponentialLR(optimizer, gamma=gamma_value)

    if scheduler_type == "StepLR":
        step_size = num_epochs/steps
        gamma_value = (final_lr/initial_lr)**(1/steps)
        return StepLR(optimizer, step_size, gamma=gamma_value)
    return None

def plot_test(model, scalers = None, function = None):
    plot_model = model.to('cpu')
    plot_model.eval()
    x_test = torch.linspace(0, 1, 100)  # Values for test x domain test
    y_test = function(x_test)  # Value for y data in test

    if scalers:
        x_scaler = scalers[0]
        y_scaler = scalers[1]
        x_test_scaled = x_scaler.transform(x_test.reshape(x_test.shape[-1], -1))
        mu, var = plot_model(x_test_scaled)  # Get predictions
    else:
        mu, var = model(x_test.reshape(x_test.shape[-1], -1))

    mu = mu.detach()  # Detach and load on cpu
    var = var.detach()
    mu_H, mu_L, mu_D = unconcatenate_mu(mu)
    var_H, var_L = unconcatenate_var(var)

    if scalers:
        #Unscale the data
        mu_H = y_scaler.inverse_transform(mu_H)
        mu_L = y_scaler.inverse_transform(mu_L)
        mu_D = y_scaler.inverse_transform_mu_D(mu_D)
        var_H = y_scaler.inverse_transform_variance(var_H)
        var_L = y_scaler.inverse_transform_variance(var_L)

    # Plot results
    fig, ax = plt.subplots(dpi=100)  # Create figure

    # Line plot for predictions and ground truth
    ax.plot(x_test, mu_H, label='H Predictions', color='green')
    ax.plot(x_test, mu_L, label='L Predictions', color='blue')
    ax.plot(x_test, mu_D, label='D Predictions', color='red')
    # Plot mu_D + mu_L
    ax.plot(x_test, mu_D + mu_L, label='mu_D + mu_L', color='purple', alpha = 0.1)
    ax.plot(x_test, y_test, label='Ground Truth', color='orange')

    # Fill the area between mu - 2*var and mu + 2*var for each prediction
    ax.fill_between(x_test, mu_H -  var_H, mu_H + var_H, color='green',alpha=0.2)
    ax.fill_between(x_test, mu_L -  var_L, mu_L + var_L, color='blue',alpha=0.2)

    # Labels and legend
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid()
    plt.title("Valid Results", fontsize=10)
    plt.show()

def pretrainer(model, hifi_data, lofi_data, valid_data, experiment_name, model_save_name, hyperparameters = None, run_name = None,device=torch.device('cpu'), n_iter=10000, learning_rate=1e-3,scheduler_type=None, save_freq=100, weight_decay= [0,0,0], valid_freq=1, batch_size = 40, num_workers = 0):
    mlflow.set_experiment(experiment_name)  # Set the name of your experiment

    # Define different weight decays
    weight_decay_mu = weight_decay[0]  # Weight decay for mu parameters
    weight_decay_var = weight_decay[1]  # Weight decay for var parameters
    weight_decay_D = weight_decay[2]  # Weight decay for mu_D parameters

    # Group mu and var parameters separately
    mu_H_params = list(model.mu_H.parameters())
    var_H_params = list(model.var_H.parameters())
    mu_L_params = list(model.mu_L.parameters())
    var_L_params = list(model.var_L.parameters())

    # Optimizers with different weight decay
    optimizer_H = optim.Adam([{'params': mu_H_params, 'weight_decay': weight_decay_mu},{'params': var_H_params, 'weight_decay': weight_decay_var}], lr=learning_rate)
    optimizer_L = optim.Adam([{'params': mu_L_params, 'weight_decay': weight_decay_mu},{'params': var_L_params, 'weight_decay': weight_decay_var}], lr=learning_rate)
    optimizer_D = optim.Adam([{'params': model.mu_D_params, 'weight_decay': weight_decay_D}], lr=learning_rate)

    scheduler_H = scheduler_case(scheduler_type, optimizer_H, learning_rate, n_iter)
    scheduler_L = scheduler_case(scheduler_type, optimizer_L, learning_rate, n_iter)
    scheduler_D = scheduler_case(scheduler_type, optimizer_D, learning_rate, n_iter)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_param("num_iter", n_iter)
        mlflow.log_param("initial_learning_rate", learning_rate)
        mlflow.log_param("schedule", scheduler_type)

        inputs_hifi, targets_hifi = hifi_data  # Extract train data
        inputs_lofi, targets_lofi = lofi_data  # Extract train data
        train_hifi_dataloader = DataLoader(list(zip(inputs_hifi, targets_hifi)), shuffle=True, batch_size=batch_size,  num_workers=num_workers)
        train_lofi_dataloader = DataLoader(list(zip(inputs_lofi, targets_lofi)), shuffle=True, batch_size=batch_size,  num_workers=num_workers)

        for iter in range(n_iter):
            model.train(True)
            # Hi-fi Epoch
            for i, train_data in enumerate(train_hifi_dataloader):
                optimizer_H.zero_grad()  # Zero gradient for every batch
                loss_hifi = torch.zeros(1, device=device)
                inputs, targets = train_data  # Extract train data
                mu_train, var_train = model(inputs[:,0].reshape(-1, 1))
                loss_hifi += loss_mse(mu_train,targets, case = 0)
                loss_hifi.backward()  # Compute the gradients and retain the graph
                optimizer_H.step()  # Take a step in the optimisation
            # Lo-fi Epoch
            for i, train_data in enumerate(train_lofi_dataloader):
                optimizer_L.zero_grad()  # Zero gradient for every batch
                loss_lofi = torch.zeros(1, device=device)
                inputs, targets = train_data  # Extract train data
                mu_train, var_train = model(inputs[:,0].reshape(-1, 1))
                loss_lofi += loss_mse(mu_train, targets, case = 1)
                loss_lofi.backward()  # Compute the gradients and retain the graph
                optimizer_L.step()  # Take a step in the optimisation
            # D Epoch
            for i, train_data in enumerate(train_hifi_dataloader):
                optimizer_D.zero_grad()  # Zero gradient for every batch
                loss_d = torch.zeros(1, device=device)
                inputs, targets = train_data  # Extract train data
                mu_train, var_train = model(inputs[:,0].reshape(-1, 1))
                loss_d += loss_mse(mu_train, targets, case = 2)
                loss_d.backward()  # Compute the gradients and retain the graph
                optimizer_D.step()  # Take a step in the optimisation

            mlflow.log_metric("train_per_epoch", loss_hifi.item() ,step=iter)
            print('Epoch: %d MSE Train H: %f' % (iter, loss_hifi.item()))

            if iter % valid_freq == 0:
                model.eval()
                with torch.no_grad():
                    valid_inputs, valid_targets = valid_data  # Extract train data
                    mu_valid, var_valid = model(valid_inputs.reshape(-1, 1))
                    valid_loss_eval = loss_mse(mu_valid.squeeze(), valid_targets.squeeze(), case=0)
                    print('Epoch: %d MSE Valid: %f' % (iter, valid_loss_eval.item()))
                mlflow.log_metric("valid_eval", valid_loss_eval.item(),step=iter)  # Print Validation loss after epoch.

            if scheduler_type is not None:
                mlflow.log_metric("current_learning_rate_both", optimizer_H.param_groups[0]['lr'], step=iter)
                scheduler_H.step()
                scheduler_L.step()
                scheduler_D.step()

            if iter % save_freq == 0:
                mlflow.pytorch.log_model(model, '%s_epoch_%f' % (model_save_name, iter))

        mlflow.pytorch.log_model(model, model_save_name)
    return run_id

def trainer(model, H_data, L_data, D_data, valid_data, experiment_name, model_save_name, hyperparameters=None, run_name=None, device=torch.device('cpu'), n_iter=10000, learning_rate=1e-3, scheduler_type=None, save_freq=100, valid_freq=1, batch_size=40, num_workers=0, tau_adjustment=True, tau=0.5, weight_decay = [0, 0, 0]):
    mlflow.set_experiment(experiment_name)  # Set the name of your experiment

    # Define different weight decays
    weight_decay_mu = weight_decay[0]  # Weight decay for mu parameters
    weight_decay_var = weight_decay[1]  # Weight decay for var parameters
    weight_decay_D = weight_decay[2]  # Weight decay for mu_D parameters

    # Group mu and var parameters separately
    mu_H_params = list(model.mu_H.parameters())
    var_H_params = list(model.var_H.parameters())
    mu_L_params = list(model.mu_L.parameters())
    var_L_params = list(model.var_L.parameters())

    # Optimizers with different weight decay
    optimizer_H = optim.Adam([{'params': mu_H_params, 'weight_decay': weight_decay_mu},{'params': var_H_params, 'weight_decay': weight_decay_var}], lr=learning_rate)
    optimizer_L = optim.Adam([{'params': mu_L_params, 'weight_decay': weight_decay_mu},{'params': var_L_params, 'weight_decay': weight_decay_var}], lr=learning_rate)
    optimizer_D = optim.Adam([{'params': model.mu_D_params, 'weight_decay': weight_decay_D}], lr=learning_rate)

    scheduler_H = scheduler_case(scheduler_type, optimizer_H, learning_rate, n_iter)
    scheduler_L = scheduler_case(scheduler_type, optimizer_L, learning_rate, n_iter)
    scheduler_D = scheduler_case(scheduler_type, optimizer_D, learning_rate, n_iter)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_param("num_iter", n_iter)
        mlflow.log_param("initial_learning_rate", learning_rate)
        mlflow.log_param("schedule", scheduler_type)

        inputs_H, targets_H = H_data  # Extract train data
        inputs_L, targets_L = L_data  # Extract train data
        inputs_D, targets_D = D_data  # Extract train data

        H_dataloader = DataLoader(list(zip(inputs_H, targets_H)), shuffle=True, batch_size=batch_size, num_workers=num_workers)
        L_dataloader = DataLoader(list(zip(inputs_L, targets_L)), shuffle=True, batch_size=batch_size, num_workers=num_workers)
        D_dataloader = DataLoader(list(zip(inputs_D, targets_D)), shuffle=True, batch_size=batch_size, num_workers=num_workers)

        for iter in range(n_iter):
            model.train(True)
            # Lo-fi Epoch
            for i, L_data in enumerate(L_dataloader):
                optimizer_L.zero_grad()  # Zero gradient for every batch
                loss_L = torch.zeros(1, device=device)
                inputs_L, targets_L = L_data  # Extract train data
                mu_train_L, var_train_L = model(inputs_L[:, 0].reshape(-1, 1))
                loss_L += loss_fn(mu_train_L, var_train_L, targets_L, hyperparameters = hyperparameters, case = 1)
                loss_L.backward()  # Compute the gradients and retain the graph
                optimizer_L.step()  # Take a step in the optimisation

            # Hi-fi Epoch
            for i, H_data in enumerate(H_dataloader):
                optimizer_H.zero_grad()  # Zero gradient for every batch
                loss_H = torch.zeros(1, device=device)
                inputs_H, targets_H = H_data  # Extract train data
                labels = inputs_H[:, -1]  # Extract labels from inputs
                mu_train_H, var_train_H = model(inputs_H[:, 0].reshape(-1, 1))
                loss_H += loss_fn(mu_train_H, var_train_H, targets_H, labels = labels, hyperparameters = hyperparameters, case = 0, tau=tau, tau_adjustment=tau_adjustment)
                loss_H.backward()  # Compute the gradients and retain the graph
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid exploding gradients
                optimizer_H.step()  # Take a step in the optimisation

            # D Epoch
            for i, D_data in enumerate(D_dataloader):
                optimizer_D.zero_grad()  # Zero gradient for every batch
                loss_D = torch.zeros(1, device=device)
                inputs_D, targets_D = D_data  # Extract train data
                mu_train_D, var_train_D = model(inputs_D[:, 0].reshape(-1, 1))
                loss_D += loss_fn(mu_train_D, var_train_D, targets_D, hyperparameters = hyperparameters, case = 2)
                loss_D.backward()  # Compute the gradients and retain the graph
                optimizer_D.step()  # Take a step in the optimisation

            mlflow.log_metric("train_NLL_per_epoch", loss_H.item(), step=iter)
            print('Epoch: %d NLL: %f' % (iter, loss_H.item()))

            if iter % valid_freq == 0:
                model.eval()
                with torch.no_grad():
                    valid_inputs, valid_targets = valid_data  # Extract train data
                    mu_valid, var_valid = model(valid_inputs.reshape(-1, 1))
                    valid_loss_eval = loss_mse(mu_valid.squeeze(), valid_targets.squeeze(), case=0)
                    print('Epoch: %d MSE Valid: %f' % (iter, valid_loss_eval.item()))
                mlflow.log_metric("valid_eval", valid_loss_eval.item(), step=iter)  # Print Validation loss after epoch.

            if scheduler_type is not None:
                mlflow.log_metric("current_learning_rate_both", optimizer_H.param_groups[0]['lr'], step=iter)
                scheduler_H.step()
                scheduler_L.step()
                scheduler_D.step()

            if iter % save_freq == 0:
                mlflow.pytorch.log_model(model, '%s_epoch_%f' % (model_save_name, iter))

        mlflow.pytorch.log_model(model, model_save_name)
    return run_id