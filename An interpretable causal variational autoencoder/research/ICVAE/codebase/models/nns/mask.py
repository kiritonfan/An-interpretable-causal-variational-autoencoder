"""
Neural network architectures for the Causal Variational Autoencoder.

This module implements the encoder and decoder networks used in the ICVAE model.
The encoder maps input data to latent representations, while the decoder reconstructs
data from the structured latent variables obtained through the DAG layer.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_paths():
    """Add project root to Python path for imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

setup_paths()

def _reduce_dimensions(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Reduce tensor dimensions by truncating extra columns."""
    return tensor[:, :target_dim]

def _expand_dimensions(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Expand tensor dimensions by zero-padding missing columns."""
    batch_size = tensor.size(0)
    device = tensor.device
    result = torch.zeros(batch_size, target_dim, device=device)
    result[:, :tensor.size(1)] = tensor
    return result

def check_dimensions(tensor: torch.Tensor, expected_dim: int, name: str = "Input") -> torch.Tensor:
    """
    Ensure tensor has the expected feature dimension.
    
    Automatically adjusts tensor dimensions by truncating or zero-padding
    to handle mismatched input dimensions gracefully.
    
    Args:
        tensor: Input tensor of shape (batch_size, features)
        expected_dim: Expected number of features
        name: Name for logging purposes
        
    Returns:
        Tensor with corrected dimensions
    """
    if tensor.size(1) != expected_dim:
        logger.warning(f"{name} dimension mismatch: Expected {expected_dim}, got {tensor.size(1)}. Adjusting.")
        if tensor.size(1) > expected_dim:
            return _reduce_dimensions(tensor, expected_dim)
        else:
            return _expand_dimensions(tensor, expected_dim)
    return tensor

class Encoder(nn.Module):
    """
    Variational encoder network for mapping geochemical element data to latent representations.
    
    The encoder takes element concentration data and outputs the parameters (mean and variance)
    of a Gaussian distribution in latent space. This follows the VAE framework for learning
    probabilistic latent representations.
    
    Architecture:
        Input (39 elements) → 1024 → 512 → 2*z_dim (mean + logvar)
        Uses ELU activations and batch normalization for stable training.
    """
    
    def __init__(self, z_dim: int = 117, num_features: int = 39):
        """
        Initialize the encoder network.
        
        Args:
            z_dim: Dimension of the latent space (typically matches total variables)
            num_features: Number of input features (typically 39 geochemical elements)
        """
        super().__init__()
        self.z_dim = z_dim
        self.num_features = num_features

        # Deep encoder network with batch normalization for stable training
        self.net = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 2 * z_dim),  # Output both mean and logvar
            nn.BatchNorm1d(2 * z_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Tuple of (mean, variance) tensors for the latent distribution,
            each of shape (batch_size, z_dim)
        """
        # Handle dimension mismatches gracefully
        if x.size(1) != self.num_features:
            x = check_dimensions(x, self.num_features, "Encoder Input")
        
        out = self.net(x)
            
        # Split output into mean and log-variance components
        mean, logvar = torch.chunk(out, 2, dim=1)
        
        # Convert log-variance to variance with numerical stability
        var = F.softplus(logvar) + 1e-6
        
        return mean, var


class Decoder_DAG(nn.Module):
    """
    Decoder network for reconstructing geochemical element data from structured latent variables.
    
    This decoder takes the structured latent variables (after causal processing through the DAG layer)
    and reconstructs the original geochemical element concentrations. The network is designed to
    handle the element variables specifically, as the causal structure typically flows from
    geological labels to element enrichment patterns.
    
    Architecture:
        Input (z_dim) → 300 → 300 → 1024 → 1024 → z2_dim (39 elements)
        Deep architecture with batch normalization for stable reconstruction.
    """
    
    def __init__(self, z_dim: int = 39, concept: int = 3, z1_dim: int = 3, z2_dim: int = 39):
        """
        Initialize the decoder network.
        
        Args:
            z_dim: Dimension of the structured latent input (typically element dimension)
            concept: Number of geological concepts/labels 
            z1_dim: Dimension of geological label variables
            z2_dim: Dimension of geochemical element variables (output dimension)
        """
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.concept = concept
        
        # Deep decoder network for element reconstruction
        self.main_decoder = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.BatchNorm1d(300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, z2_dim),  # Output geochemical element concentrations
            nn.BatchNorm1d(z2_dim)
        )
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode structured latent variables back to element concentrations.
        
        Args:
            z: Structured latent tensor of shape (batch_size, z_dim)
               representing element variables after causal processing
               
        Returns:
            Reconstructed element concentrations of shape (batch_size, z2_dim)
        """
        batch_size = z.size(0)
        
        # Ensure proper tensor shape for decoder input
        if z.dim() > 2 or z.size(1) != self.z_dim:
            z = z.contiguous().view(batch_size, self.z_dim)
        
        # Reconstruct element concentrations through deep network
        output = self.main_decoder(z)
        return output
