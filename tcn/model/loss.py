import torch
import numpy as np
from torch import nn

def max_sharpe(y_return, weights, rf=0.0524, device="cuda", lb=0.01, ub=0.9, K=10, smoothing_factor=50.0):
    weights = torch.unsqueeze(weights, 1)
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
    
    # Handle single-day returns case
    if y_return.shape[1] <= 1:
        # Create a diagonal covariance matrix with small values
        batch_size = y_return.shape[0]
        n_assets = y_return.shape[2]
        # Use a small variance value (0.01) for stability
        covmat = torch.eye(n_assets).unsqueeze(0).repeat(batch_size, 1, 1).to(device) * 0.01
    else:
        # Original computation for multi-day returns
        covmat = torch.Tensor(np.array([np.cov(batch.cpu().T, ddof=0) for batch in y_return])).to(device)
    
    portReturn = torch.matmul(weights, meanReturn)
    portVol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    objective = (portReturn * 252 - rf) / (torch.sqrt(portVol * 252))

    # Add boundary constraint penalties
    weights_flat = weights.squeeze(1)
    
    # Maximum weight penalty - differentiable with ReLU
    max_weight_penalty = torch.mean(torch.relu(weights_flat - ub)) * 2.0
    
    # Minimum weight penalty - use smooth version
    # Instead of (weights_flat > 0).float(), use sigmoid for smooth transition
    weight_presence = torch.sigmoid(smoothing_factor * weights_flat)
    min_weight_penalty = torch.mean(torch.relu(lb - weights_flat) * weight_presence) * 1.0
    
    # For cardinality constraint - avoid sorting
    # We want to encourage K largest weights and discourage others
    
    # Compute soft count of active assets (differentiable)
    active_assets = torch.sum(weight_presence, dim=1)
    
    # Penalize having too few assets (differentiable with ReLU)
    too_few_penalty = torch.mean(torch.relu(K - active_assets)) * 1.0
    
    # For too many assets penalty, use L1 regularization with a threshold
    # This encourages sparsity more smoothly than sorting
    # Sort weights for visualization purposes only (not used in gradient)
    _, indices = torch.sort(weights_flat, dim=1, descending=True)
    
    # Create a penalty coefficient tensor that's 0 for top-K weights, 2.0 for others
    # Using smooth approximation of step function
    rank = torch.arange(weights_flat.shape[1], device=device).float().unsqueeze(0)
    rank = rank.expand(weights_flat.shape[0], -1)
    # Smooth step function: sigmoid((rank-K+0.5)*factor)
    penalty_coef = torch.sigmoid((rank - K + 0.5) * smoothing_factor) * 1.0
    
    # Apply penalty coefficients element-wise
    too_many_penalty = torch.mean(torch.sum(weights_flat * penalty_coef, dim=1))
    
    # Combine all penalties
    total_penalty = max_weight_penalty + min_weight_penalty + too_many_penalty + too_few_penalty

    return -objective.mean() + total_penalty

def equal_risk_parity(y_return, weights):
    B = y_return.shape[0]
    F = y_return.shape[2]
    weights = torch.unsqueeze(weights, 1).to("cuda")
    covmat = torch.Tensor(
        [np.cov(batch.cpu().T, ddof=0) for batch in y_return]
    )  # (batch, 50, 50)
    covmat = covmat.to("cuda")
    sigma = torch.sqrt(
        torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    )
    mrc = (1 / sigma) * (covmat @ torch.transpose(weights, 2, 1))
    rc = weights.view(B, F) * mrc.view(B, F)
    target = (torch.ones((B, F)) * (1 / F)).to("cuda")
    risk_diffs = rc - target
    sum_risk_diffs_squared = torch.mean(torch.square(risk_diffs))
    return sum_risk_diffs_squared


if __name__ == "__main__":
    pass
