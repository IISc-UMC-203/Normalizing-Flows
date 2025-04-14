# src/losses.py
import torch
import math
import numpy as np

PI = torch.tensor(math.pi)

def nll_loss_2d(z, log_det_total):
    """Negative Log Likelihood loss for 2D RealNVP."""
    # Assumes base distribution p_Z is N(0, I)
    # log p_X(x) = log p_Z(f(x)) + log |det(df/dx)|
    log_pz = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.shape[1] * torch.log(2 * PI)
    log_px = log_pz + log_det_total
    nll = -torch.mean(log_px)
    return nll

def nll_loss_img(y, s_log_det_flat, norm_log_det_flat, scale_params_flat, batch_size, preprocess_log_det_sum=None):
    num_pixels = y.shape[1] # Total number of dimensions

    # Log probability of latent variable under N(0, I) prior
    log_pz = -0.5 * torch.log(2 * PI) * num_pixels - 0.5 * torch.sum(y**2, dim=1) # Sum over spatial/channel dims

    # Log determinant from coupling layers
    log_det_s = torch.sum(s_log_det_flat, dim=1) # Sum over all coupling contributions

    # Log determinant from BatchNorm layers
    log_det_norms = torch.sum(norm_log_det_flat, dim=1) 
    reg = 5e-5 * torch.sum(scale_params_flat ** 2)
    log_px = log_pz + log_det_s + log_det_norms

    # Account for preprocessing log determinant if provided
    if preprocess_log_det_sum is not None:
        log_px = log_px + preprocess_log_det_sum

    # Average NLL over batch
    nll = -torch.mean(log_px)

    # Add regularization
    loss = nll + reg / batch_size # Average regularization effect over batch

    # Return total loss and components for logging (using mean values over batch)
    log_px_mean = -torch.mean(log_pz)
    det_mean = torch.mean(log_det_s)
    norms_mean = torch.mean(log_det_norms)
    reg_mean = reg # Regularization is total sum

    return loss, (log_px_mean, det_mean, norms_mean, reg_mean)