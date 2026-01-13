import torch

def generate_hifi_lofi_data(hifi_total_num_points, hifi_regions, hifi_regions_num_points, hifi_regions_noise_mu, hifi_regions_noise_stds, function, lofi_function, std_lofi = 0.01):
    start_idx = 0
    x_hifi_data = torch.zeros((hifi_total_num_points, 2))  # Empty data sets
    y_hifi_data = torch.zeros((hifi_total_num_points,))  # Empty data sets
    x_lofi_match_data = torch.zeros((hifi_total_num_points, 2))  # Empty data sets
    y_lofi_match_data = torch.zeros((hifi_total_num_points,))  # Empty data sets
    for i, (region, num_points, mu, std) in enumerate(zip(hifi_regions, hifi_regions_num_points, hifi_regions_noise_mu, hifi_regions_noise_stds)):
        # Generate data for the region
        x = (region[0] - region[1]) * torch.rand(num_points) + region[1]
        y_hifi = function(x) + torch.normal(mean=mu, std=std, size=(num_points,))
        y_lofi = lofi_function(x) + torch.normal(mean=0, std=std_lofi, size=(num_points,))

        # Store the data
        end_idx = start_idx + num_points
        x_hifi_data[start_idx:start_idx + num_points, 0] = x  # Store x values
        x_hifi_data[start_idx:start_idx + num_points, 1] = 0
        y_hifi_data[start_idx:start_idx + num_points] = y_hifi  # Store noisy y values
        x_lofi_match_data[start_idx:start_idx + num_points, 0] = x
        x_lofi_match_data[start_idx:start_idx + num_points, 1] = 1
        y_lofi_match_data[start_idx:start_idx + num_points] = y_lofi
        start_idx = end_idx
    return x_hifi_data, y_hifi_data, x_lofi_match_data, y_lofi_match_data

def generate_lofi_data(lofi_total_num_points, lofi_regions, lofi_regions_num_points, lofi_function, std_lofi = 0.01):
    # Create empty training, validation data
    start_idx = 0  # Restart index
    x_lofi_only_data = torch.zeros((lofi_total_num_points, 2))  # Empty data sets
    y_lofi_only_data = torch.zeros((lofi_total_num_points,))  # Empty data sets
    for i, (region, num_points) in enumerate(zip(lofi_regions, lofi_regions_num_points)):
        # Generate data for the region
        x = torch.linspace(region[0], region[1], num_points)
        y = lofi_function(x) + torch.normal(mean=0, std=std_lofi, size=(num_points,))
        # Store the data
        end_idx = start_idx + num_points
        x_lofi_only_data[start_idx:end_idx, 0] = x  # Store x values
        x_lofi_only_data[start_idx:end_idx, 1] = 1
        y_lofi_only_data[start_idx:end_idx] = y  # Store noisy y values
        start_idx = end_idx
    return x_lofi_only_data, y_lofi_only_data

def generate_pretrain_lofi_data(pre_train_lofi_total_num_points, pre_train_lofi_regions, pre_train_lofi_num_points, lofi_function, std_lofi = 0.01):
    start_idx = 0  # Restart index
    x_lofi_pretrain_data = torch.zeros((pre_train_lofi_total_num_points, 2))  # Empty data sets
    y_lofi_pretrain_data = torch.zeros((pre_train_lofi_total_num_points,))  # Empty data sets
    for i, (region, num_points) in enumerate(zip(pre_train_lofi_regions, pre_train_lofi_num_points)):
        # Generate data for the region
        x = torch.linspace(region[0], region[1], num_points)
        y = lofi_function(x) + torch.normal(mean=0, std=std_lofi, size=(num_points,))
        # Store the data
        end_idx = start_idx + num_points
        x_lofi_pretrain_data[start_idx:end_idx,0] = x  # Store x values
        x_lofi_pretrain_data[start_idx:end_idx, 1] = 1
        y_lofi_pretrain_data[start_idx:end_idx] = y  # Store noisy y values
        start_idx = end_idx
    return x_lofi_pretrain_data, y_lofi_pretrain_data

def generate_test_data(test_regions, test_regions_num_points, function):
    x_test_data = torch.linspace(test_regions[0], test_regions[1], test_regions_num_points)  # Values for x domain test
    y_test_data = torch.tensor(function(x_test_data))  # Value for y data in test
    return x_test_data, y_test_data