import torch

def unconcatenate_mu(x):
    x_H = x[:, 0]
    x_L = x[:, 1]
    x_D = x[:, 2]
    return x_H, x_L, x_D

def unconcatenate_var(x):
    x_H = x[:, 0]
    x_L = x[:, 1]
    return x_H, x_L

def beta_calculator(alpha, mode):
    beta = (alpha + 1) * mode
    return beta

def desired_ratio_var_H_overlap(r, alpha):
    return 4 * r * (alpha + 1)

def desired_var_0_mu_H_overlap(r, n_H, var_H):
    return var_H * 2 * r / n_H

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before transforming.")
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before inverse transforming.")
        return x * self.std + self.mean

    def inverse_transform_mu_D(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before inverse transforming.")
        return x * self.std

    def inverse_transform_variance(self, x):
        return x * self.std ** 2

def create_scalers(x_data, y_data):
    # Scale data using a single scaler for all the data.
    x_Global_Scaler = Scaler()
    y_Global_Scaler = Scaler()

    # Fit to combined data
    x_Global_Scaler.fit(x_data[:, 0].squeeze())
    y_Global_Scaler.fit(y_data)

    # Store scalers
    scalers = [x_Global_Scaler, y_Global_Scaler]
    return scalers

def scale_data(x_data, y_data, scalers, label = True):
    if label == True:
        x_scaled = scalers[0].transform(x_data[:, 0].squeeze())
        x_scaled = torch.stack((x_scaled, x_data[:, 1].squeeze()), dim = 1)  # Keep other features intact
        y_scaled = scalers[1].transform(y_data)
    else:
        x_scaled = scalers[0].transform(x_data)
        y_scaled = scalers[1].transform(y_data)
    return x_scaled, y_scaled