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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

setup_paths()

def _reduce_dimensions(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    return tensor[:, :target_dim]

def _expand_dimensions(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    batch_size = tensor.size(0)
    device = tensor.device
    result = torch.zeros(batch_size, target_dim, device=device)
    result[:, :tensor.size(1)] = tensor
    return result

def check_dimensions(tensor: torch.Tensor, expected_dim: int, name: str = "Input") -> torch.Tensor:
    if tensor.size(1) != expected_dim:
        logger.warning(f"{name} dimension mismatch: Expected {expected_dim}, got {tensor.size(1)}. Adjusting.")
        if tensor.size(1) > expected_dim:
            return _reduce_dimensions(tensor, expected_dim)
        else:
            return _expand_dimensions(tensor, expected_dim)
    return tensor

class Encoder(nn.Module):
    def __init__(self, z_dim: int = 117, num_features: int = 39):
        super().__init__()
        self.z_dim = z_dim
        self.num_features = num_features

        self.net = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 2 * z_dim),
            nn.BatchNorm1d(2 * z_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) != self.num_features:
            x = check_dimensions(x, self.num_features, "Encoder Input")
        
        out = self.net(x)
            
        mean, logvar = torch.chunk(out, 2, dim=1)
        
        var = F.softplus(logvar) + 1e-6
        
        return mean, var


class Decoder_DAG(nn.Module):
    def __init__(self, z_dim: int = 39, concept: int = 3, z1_dim: int = 3, z2_dim: int = 39):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.concept = concept
        
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
            nn.Linear(1024, z2_dim),
            nn.BatchNorm1d(z2_dim)
        )
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        if z.dim() > 2 or z.size(1) != self.z_dim:
            z = z.contiguous().view(batch_size, self.z_dim)
        
        output = self.main_decoder(z)
        return output