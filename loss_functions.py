import torch

def unconcatenate_mu(x):
    x_H = x[:, 0]
    x_L = x[:, 1]
    x_0 = x[:, 2]
    return x_H, x_L, x_0

def unconcatenate_var(x):
    x_H = x[:, 0]
    x_L = x[:, 1]
    return x_H, x_L

def loss_fn(mu, var, targets, labels = None, hyperparameters = None, case = 0, tau_adjustment = False, tau = 0.5):
    mu_H, mu_L, mu_D= unconcatenate_mu(mu)
    var_H, var_L = unconcatenate_var(var)

    # Create a mask for where labels are - 0 or 1
    if labels is not None:
        mask_H = (labels == 0).squeeze()

    # Detached variables
    mu_H_detached = mu_H.detach()
    mu_L_detached = mu_L.detach()
    mu_D_detached = mu_D.detach()
    mu_L_D_detached = (mu_L_detached + mu_D_detached)
    mu_H_L_detached = (mu_H_detached - mu_L_detached)

    if hyperparameters is not None:
        var_0, alpha_0, beta_0, n_count = hyperparameters

    if case == 0:  # H case: likelihood function with prior on mu_H - mu_L and var_H
        eps = 1e-6  # Small epsilon to prevent log(0) or division by 0

        # Clamp var_H for stability
        var_H_clamped = torch.clamp(var_H, min=eps)
        var_H_masked_clamped = torch.clamp(var_H[mask_H], min=eps)

        # Log-likelihood for H (only on high-fidelity data)
        H_log_likelihood =- (n_count[0] / 2) * torch.log(var_H_masked_clamped)
        H_log_likelihood -= (n_count[0] / 2) * ((mu_H[mask_H] - targets[mask_H]) ** 2) / var_H_masked_clamped

        # Prior on mu_D (applies to all data, not masked)
        D_log_prior = -0.5 * ((mu_H - mu_L_D_detached) ** 2) / var_0

        # Prior on var_H (entire vector)
        H_var_log_prior =- (alpha_0 + 1) * torch.log(var_H_clamped)
        H_var_log_prior -= beta_0 / var_H_clamped

        # Optional tau adjustment (multiplicative weighting)
        if tau_adjustment:
            # Clamp before raising to a power to avoid overflow/underflow
            var_H_adj = torch.clamp(var_H[mask_H].detach(), min=eps, max=1e6)
            adjustment = var_H_adj ** tau
            H_log_likelihood = adjustment * H_log_likelihood

            var_H_all_adj = torch.clamp(var_H.detach(), min=eps, max=1e6)
            adjustment_all = var_H_all_adj ** tau
            H_var_log_prior = adjustment_all * H_var_log_prior

        return - torch.mean(H_log_likelihood) - torch.mean(D_log_prior) - torch.mean(H_var_log_prior)

    if case == 1: # L case: likelihood function with prior on mu_H - mu_L and var_H
        # Log-likelihoods for L
        L_log_likelihood = -(n_count[1] / 2) * torch.log(var_L) - (n_count[1] / 2) * (mu_L - targets) ** 2 / (var_L)
        if tau_adjustment == True:
            L_log_likelihood = var_H.detach() ** (tau) * L_log_likelihood
        return -torch.mean(L_log_likelihood)

    if case == 2: # D case: likelihood function with prior on mu_H - mu_L and var_H
        # D log prior
        D_log_prior = - (1/2) * (mu_D - (mu_H_L_detached)) ** 2 / (var_0)
        return -torch.mean(D_log_prior)

def loss_mse(mu, targets, case = 0):
    mu_H, mu_L, mu_D = unconcatenate_mu(mu)
    mu_H_detached = mu_H.detach()
    mu_L_detached = mu_L.detach()
    mu_H_L_detached = (mu_H_detached - mu_L_detached)
    if case == 0:
        se_H = (mu_H - targets) ** 2
        return torch.mean(se_H)
    if case == 1:
        se_L = (mu_L - targets) ** 2
        return torch.mean(se_L)
    if case == 2:
        se_D = (mu_D - mu_H_L_detached) ** 2
        return torch.mean(se_D)

def logmeanexp(inputs, dim=0):
    input_max = inputs.max(dim=dim)[0]
    return (inputs - input_max).exp().mean(dim=dim).log() + input_max