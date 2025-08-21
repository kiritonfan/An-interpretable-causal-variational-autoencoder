"""
ICVAE Unified Utilities
======================

All utility functions for the ICVAE model in one clean file.
Includes: mathematical operations, DAG constraints, and helper functions.

Functions:
- sample_gaussian, kl_normal: VAE mathematical operations
- compute_dag_constraint: DAG acyclicity constraint for causal learning
"""

import torch
import numpy as np


# =============================================================================
# VAE MATHEMATICAL UTILITIES
# =============================================================================

def sample_gaussian(m, v):
    """Sample from Gaussian using reparameterization trick: z = m + sqrt(v) * eps"""
    v = torch.clamp(v, min=1e-8)  # Numerical stability
    epsilon = torch.randn_like(m)
    z = m + torch.sqrt(v) * epsilon
    return z


def kl_normal(qm, qv, pm, pv):
    """Compute KL divergence between two normal distributions: KL[q || p]"""
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv/pv + (qm-pm).pow(2)/pv - 1)
    element_wise = torch.clamp(element_wise, min=1e-8, max=1e8)  # Numerical stability
    
    if element_wise.dim() > 1:
        kl = element_wise.sum(dim=-1).mean()
    else:
        kl = element_wise.mean()
    
    if kl.dim() > 0:
        kl = kl.mean()
        
    return kl


# =============================================================================
# DAG CONSTRAINT UTILITIES
# =============================================================================

def compute_dag_constraint(A: torch.Tensor, m: int) -> torch.Tensor:
    """
    Compute DAG acyclicity constraint: h(A) = tr(exp(A ⊙ A)) - d
    Returns 0 if A represents a valid DAG (no cycles).
    """
    d = m
    
    # Remove diagonal elements to prevent self-loops
    A_no_diag = A * (1 - torch.eye(d, device=A.device))
    
    # Compute matrix exponential using polynomial approximation for stability
    eps = 1e-6
    M = torch.eye(d, device=A.device) + A_no_diag/d + eps * torch.eye(d, device=A.device)
    
    # Matrix power computation: (I + A/d)^d ≈ exp(A)
    E = torch.matrix_power(M, d)
    h_A = torch.trace(E) - d
    
    # Ensure scalar output for gradient computation
    if h_A.dim() > 0:
        h_A = h_A.mean()
        
    # Numerical stability: prevent negative values due to floating point errors
    if h_A < 1e-8:
        h_A = torch.tensor(1e-8, device=A.device)
        
    return h_A
