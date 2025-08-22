"""
Causal Variational Autoencoder (CausalVAE) implementation.

Core model architecture integrating variational autoencoders with directed
acyclic graph (DAG) constraints for causal structure learning in geochemical data.

Key components:
- CausalVAE: Main model combining encoder, decoder, and DAG layer
- DagLayer: Implements structural causal constraints via adjacency matrix
- Conditional prior network: Label-conditioned latent variable generation
- Loss functions: ELBO + DAG acyclicity + sparsity constraints
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

setup_paths()

from codebase import utils as ut
from codebase.models.nns import mask

def _get_optimal_device() -> torch.device:
    """Select best available GPU device based on memory capacity."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return torch.device("cuda:0")
    
    # Choose GPU with maximum memory for complex causal models
    max_memory = 0
    best_device_index = 0
    for i in range(device_count):
        memory = torch.cuda.get_device_properties(i).total_memory
        if memory > max_memory:
            max_memory = memory
            best_device_index = i
    return torch.device(f"cuda:{best_device_index}")


class ModelConfig:
    """Global configuration for model training and numerical stability."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42          # Reproducibility seed
        self.eps = 1e-6         # Numerical stability epsilon

    def set_seed(self):
        """Set random seeds for reproducible training."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

config = ModelConfig()
config.set_seed()

class CausalVAE(nn.Module):
    """
    Causal Variational Autoencoder for learning structured representations.
    
    Integrates VAE with DAG constraints to discover causal relationships between
    geological labels and geochemical elements. The model enforces structural
    causality through learnable adjacency matrices while maintaining VAE's
    generative capabilities.
    
    Architecture:
        Input → Encoder → Latent(z) → DAG Layer → Structured(z') → Decoder → Output
        
    Key features:
    - Conditional prior based on geological labels
    - DAG-constrained causal structure learning  
    - Interpretable causal relationship extraction
    - Multi-loss training with ELBO + DAG penalties
    """
    
    def __init__(self, 
                 nn_type: str = 'mask', 
                 name: str = 'vae',
                 z_dim: int = 117,      # Total VAE latent dimension
                 z1_dim: int = 3,       # Geological label dimension
                 z2_dim: int = 39,      # Geochemical element dimension
                 concept: int = 3,      # Number of concept categories
                 element_relations: bool = True, 
                 initial: bool = True,
                 device: Optional[torch.device] = None):
        """
        Initialize CausalVAE with specified architecture and dimensions.
        
        Args:
            z_dim: VAE latent space dimension (typically z1_dim + z2_dim + extra)
            z1_dim: Number of geological label variables
            z2_dim: Number of geochemical element variables  
            concept: Number of geological concept categories
            element_relations: Whether to learn direct label-element relations
            initial: Whether to initialize parameters with Xavier initialization
        """
        super().__init__()
        
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.concept = concept
        self.nn_type = nn_type
        self.element_relations = element_relations
        self.device = device if device else _get_optimal_device()
        
        self._validate_dimensions()
        self._init_components(initial)
        
        self.to(self.device)
        logger.info(f"CausalVAE initialized: z_dim(VAE)={z_dim}, z1_dim(labels)={z1_dim}, z2_dim(elements)={z2_dim}")

    def _validate_dimensions(self):
        """Validate model dimension consistency and warn about potential issues."""
        if self.z1_dim <= 0 or self.z2_dim <= 0:
            raise ValueError("Label and element dimensions must be positive.")
        if self.z_dim < self.z2_dim:
            logger.warning(f"VAE latent dimension ({self.z_dim}) is less than element dimension ({self.z2_dim}), which might be unintended.")

    def _init_components(self, initial: bool):
        """Initialize all model components: encoder, decoder, DAG layer, and prior network."""
        # VAE encoder: elements → latent distribution parameters
        self.encoder = mask.Encoder(z_dim=self.z_dim, num_features=self.z2_dim)
        
        # Project full latent space to element-specific latent space
        self.project_z_to_elements = nn.Linear(self.z_dim, self.z2_dim)
        
        # VAE decoder: structured latent → reconstructed elements
        self.decoder = mask.Decoder_DAG(
            z_dim=self.z2_dim,
            concept=self.concept, 
            z1_dim=self.z1_dim,
            z2_dim=self.z2_dim
        )
        
        # DAG layer: enforces causal structure constraints
        self.dag = DagLayer(
            z1_dim=self.z1_dim, 
            z2_dim=self.z2_dim, 
            initial=initial
        )
        
        self._init_prior_network()
        
        if self.element_relations:
            self._init_element_relations()

    def _init_prior_network(self):
        """
        Build conditional prior network: labels → VAE latent distribution parameters.
        
        Maps geological labels to mean and variance of latent distribution,
        enabling label-conditional generation and improved ELBO bounds.
        """
        self.prior_network = nn.Sequential(
            nn.Linear(self.z1_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2 * self.z_dim)  # Output: mean + logvar for latent prior
        )
        
        # Conservative initialization for stable prior learning
        for module in self.prior_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def _init_element_relations(self):
        """Optional direct parameterization for label↔element relations.

        These parameters are not directly optimized in the loss but serve as
        interpretable hooks for inspecting direct associations in addition to
        the DAG-induced structure.
        """
        self.label_to_element = nn.Parameter(torch.zeros(self.z1_dim, self.z2_dim))
        self.element_to_label = nn.Parameter(torch.zeros(self.z2_dim, self.z1_dim))
        
        nn.init.xavier_uniform_(self.label_to_element, gain=0.1)
        nn.init.xavier_uniform_(self.element_to_label, gain=0.1)
        
        self.register_buffer('causal_strength', torch.zeros(self.z1_dim, self.z2_dim))

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode → sample → DAG transform → decode.
        
        Core CausalVAE inference combining variational encoding with structural
        causal processing through the DAG layer.
        
        Args:
            x: Input geochemical element data [batch_size, z2_dim]
            label: Geological labels [batch_size, z1_dim]
            
        Returns:
            Dictionary containing reconstruction and distribution parameters
        """
        x = x.to(self.device)
        label = self._ensure_tensor(label).to(self.device)
        
        # VAE encoding: elements → latent distribution parameters
        q_m_raw, q_v_raw_unactivated = self.encoder.encode(x)
        q_v_raw = F.softplus(q_v_raw_unactivated) + config.eps  # Ensure positive variance
        z_raw = ut.sample_gaussian(q_m_raw, q_v_raw)

        # Project latent sample to element-specific noise term
        element_noise = self.project_z_to_elements(z_raw)
        
        # Combine labels with element noise for DAG processing
        dag_exogenous_input = torch.cat((label, element_noise), dim=1)

        # Apply DAG transformation: (I - A)^(-1) * input
        z_structured = self.dag(dag_exogenous_input)
        
        # Handle root nodes (no incoming edges) - use original input
        with torch.no_grad():
            A = self.dag.A
            is_root_node_mask = torch.sum(torch.abs(A), dim=1) == 0
        z_for_decoder_input = torch.where(is_root_node_mask, dag_exogenous_input, z_structured)

        # Extract element portion for reconstruction
        z_elements_for_decoding = z_for_decoder_input[:, self.z1_dim:]
        
        # Decode structured latent variables to element concentrations
        x_recon = self.decoder.decode(z_elements_for_decoding)
        
        # Get label-conditional prior parameters
        p_m_raw, p_v_raw = self.get_conditional_prior_params(label)
        
        return {
            'x_recon': x_recon,
            'q_m_raw': q_m_raw,
            'q_v_raw': q_v_raw,
            'p_m_raw': p_m_raw,
            'p_v_raw': p_v_raw,
        }

    def negative_elbo_bound(self, 
                          x: torch.Tensor, 
                          label: torch.Tensor,
                          use_conditional_prior: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute negative Evidence Lower BOund (ELBO) for VAE training.
        
        ELBO = E[log p(x|z)] - KL[q(z|x) || p(z|label)]
        
        Returns:
            Tuple of (negative_ELBO, KL_divergence, reconstruction_loss)
        """
        try:
            outputs = self.forward(x, label)
            x_hat = outputs['x_recon']
            q_m_raw, q_v_raw = outputs['q_m_raw'], outputs['q_v_raw']
            
            # Use conditional or standard prior based on flag
            p_m_raw, p_v_raw = (outputs['p_m_raw'], outputs['p_v_raw']) if use_conditional_prior else \
                               (torch.zeros_like(q_m_raw), torch.ones_like(q_v_raw))
            
            # KL divergence: posterior vs prior
            kl_raw = ut.kl_normal(q_m_raw, q_v_raw, p_m_raw, p_v_raw)
            kl_raw = self._ensure_scalar(kl_raw.sum(dim=-1))

            # Reconstruction loss: MSE between input and output
            x_hat = self._match_tensor_shapes(x, x_hat)
            rec = F.mse_loss(x_hat, x, reduction='mean')
            
            nelbo = rec + kl_raw
            
            return nelbo, kl_raw, rec
            
        except Exception as e:
            logger.error(f"Error calculating ELBO: {e}", exc_info=True)
            safe_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            return safe_loss, safe_loss, safe_loss

    def loss(self, 
             x: torch.Tensor, 
             label: torch.Tensor,
             rec_weight: float = 1.0,
             kl_weight: float = 0.5,
             dag_weight: float = 0.1,
             use_conditional_prior: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Complete training loss: ELBO + DAG constraints.
        
        Combines variational objective with structural causal penalties:
        - Reconstruction loss: data fidelity
        - KL loss: regularization 
        - DAG acyclicity: prevent cycles in causal graph
        - Sparsity: encourage sparse causal connections
        """
        _, kl_raw, rec = self.negative_elbo_bound(x, label, use_conditional_prior=use_conditional_prior)
        
        # Compute DAG-specific penalty terms
        dag_loss_structural, sparsity_loss_A = self._compute_dag_losses()
        
        # Ensure minimum DAG weight for structural learning
        dag_weight = max(0.01, dag_weight)
        
        # Weighted combination of all loss terms
        total_loss = (rec_weight * rec + 
                      kl_weight * kl_raw + 
                      dag_weight * dag_loss_structural + 
                      sparsity_loss_A)
        
        # Safety check for numerical stability
        if torch.isnan(total_loss):
            logger.warning("NaN loss detected, replacing with a safe value.")
            total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        summaries = self._create_loss_summaries(
            total_loss, rec, kl_raw, dag_loss_structural, sparsity_loss_A, 
            rec_weight, kl_weight, dag_weight
        )
        
        return total_loss, summaries

    def _compute_dag_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DAG constraint penalties: acyclicity and sparsity.
        
        Returns:
            acyclicity_loss: Penalty for cycles in the graph (h(A) constraint)
            sparsity_loss: L1 penalty encouraging sparse adjacency matrix
        """
        dag_param = self.dag.A
        
        # Acyclicity constraint: h(A) = tr(e^(A◦A)) - d should be 0 for DAGs
        h_a = self._h_A_robust(dag_param, dag_param.shape[0])
        h_a = torch.clamp(h_a, min=config.eps)  # Avoid negative values
        dag_loss_structural = h_a + 0.5 * h_a * h_a  # Quadratic penalty
        
        # Sparsity penalty: encourage few causal connections
        sparsity_loss_A = 0.05 * torch.sum(torch.abs(dag_param))
        
        return self._ensure_scalar(dag_loss_structural), self._ensure_scalar(sparsity_loss_A)

    def _create_loss_summaries(self, total_loss, rec, kl_raw, dag_structural, sparsity_A, 
                             rec_w, kl_w, dag_w) -> Dict[str, float]:
        return {
            'train/loss_total': self._to_float(total_loss),
            'train/rec_loss': self._to_float(rec),
            'train/kl_raw_loss': self._to_float(kl_raw),
            'train/dag_structural_loss': self._to_float(dag_structural),
            'train/dag_sparsity_loss_A': self._to_float(sparsity_A),
            'train/rec_weighted': self._to_float(rec_w * rec),
            'train/kl_raw_weighted': self._to_float(kl_w * kl_raw),
            'train/dag_structural_weighted': self._to_float(dag_w * dag_structural),
        }

    def get_conditional_prior_params(self, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given labels, return mean and variance of the conditional prior p(z|label)."""
        label = self._ensure_tensor(label)
        if label.dim() == 1: label = label.unsqueeze(0)
        if label.size(0) == 0: return torch.empty(0, self.z_dim, device=self.device), torch.empty(0, self.z_dim, device=self.device)

        prior_params = self.prior_network(label)
        mean, logvar = torch.chunk(prior_params, 2, dim=1)
        var = F.softplus(logvar) + config.eps
        return mean, var

    def sample_z_conditional(self, batch_size: int, label: torch.Tensor) -> torch.Tensor:
        label = self._ensure_tensor(label)
        if label.dim() == 1: label = label.unsqueeze(0)
        if label.size(0) == 0: return torch.empty(0, self.z_dim, device=self.device)

        p_m_raw, p_v_raw = self.get_conditional_prior_params(label)
        epsilon = torch.randn_like(p_m_raw)
        
        return p_m_raw + torch.sqrt(p_v_raw) * epsilon

    def reconstruct(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Reconstruct inputs given labels by passing through encoder, DAG, and decoder."""
        return self.forward(x, label)['x_recon']

    def get_dag_params(self) -> torch.Tensor:
        """Return a copy of the learned DAG adjacency with zeroed diagonal."""
        dag_matrix = self.dag.A.detach().clone()
        dag_matrix.fill_diagonal_(0)
        return dag_matrix

    def get_causal_relations(self) -> Dict[str, np.ndarray]:
        dag_A = self.get_dag_params()
        relations = {
            'dag_A_matrix': dag_A.cpu().numpy(),
            'label_to_element_dag': dag_A[:self.z1_dim, self.z1_dim:].cpu().numpy(),
            'element_to_label_dag': dag_A[self.z1_dim:, :self.z1_dim].cpu().numpy(),
            'element_to_element_dag': dag_A[self.z1_dim:, self.z1_dim:].cpu().numpy(),
            'label_to_label_dag': dag_A[:self.z1_dim, :self.z1_dim].cpu().numpy()
        }
        if self.element_relations:
            relations['direct_label_to_element_param'] = self.label_to_element.detach().cpu().numpy()
            relations['direct_element_to_label_param'] = self.element_to_label.detach().cpu().numpy()
        return relations

    def perform_do_operation(self, interventions: Dict[str, float], all_names: List[str], batch_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a simple do-intervention on the structural model and decode outputs."""
        self.eval()
        with torch.no_grad():
            n = torch.randn(batch_size, self.dag.total_dim, device=self.device)

            for var_name, value in interventions.items():
                try:
                    idx = all_names.index(var_name)
                    n[:, idx] = value
                except ValueError:
                    logger.warning(f"Variable '{var_name}' not found in intervention list, skipping.")
            
            z_structured = self.dag(n)

            z_elements_for_decoding = z_structured[:, self.z1_dim:]
            reconstructed_output = self.decoder.decode(z_elements_for_decoding)

            return reconstructed_output, z_structured

    def _h_A_robust(self, A: torch.Tensor, m: int) -> torch.Tensor:
        return ut.compute_dag_constraint(A, m)

    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        if not torch.is_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        return x.to(self.device)

    def _ensure_scalar(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean() if x.dim() > 0 else x

    def _to_float(self, x: torch.Tensor) -> float:
        return x.item()

    def _match_tensor_shapes(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if target.shape != source.shape:
            logger.warning(f"Shape mismatch in reconstruction: target {target.shape}, source {source.shape}. Attempting to reshape source.")
            try:
                return source.view(target.shape)
            except RuntimeError as e:
                logger.error(f"Cannot reshape source to target shape: {e}")
        return source


class DagLayer(nn.Module):
    """
    Directed Acyclic Graph (DAG) layer for structural causal modeling.
    
    Implements the core causal transformation z = (I - A)^(-1) * ε where:
    - A is a learnable adjacency matrix (with zero diagonal)
    - ε are exogenous noise variables (labels + element noise)
    - z are the resulting structured variables after causal processing
    
    The DAG constraint ensures A represents a valid causal structure without cycles.
    """
    
    def __init__(self, z1_dim: int = 3, z2_dim: int = 39, initial: bool = True):
        """
        Initialize DAG layer with specified variable dimensions.
        
        Args:
            z1_dim: Number of geological label variables
            z2_dim: Number of geochemical element variables  
            initial: Whether to apply Xavier initialization to adjacency matrix
        """
        super().__init__()
        
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.total_dim = z1_dim + z2_dim  # Combined variable space
        self.device = _get_optimal_device()

        # Learnable adjacency matrix A[i,j] = causal effect from j → i
        self.A = nn.Parameter(torch.zeros(self.total_dim, self.total_dim, device=self.device))
        
        if initial:
            self._initialize_parameters()
        
        logger.info(f"DAG layer initialized: total_dim={self.total_dim} on device {self.device}")

    def _initialize_parameters(self):
        """Xavier-initialize adjacency and zero the diagonal to avoid self-loops."""
        nn.init.xavier_uniform_(self.A, gain=0.1)
        with torch.no_grad():
            self.A.fill_diagonal_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute (I - A)^-1 x where A is masked to exclude diagonal.

        Falls back to pseudo-inverse if direct inversion fails for numerical reasons.
        """
        batch_size = x.size(0)
        
        if x.size(1) != self.total_dim:
            logger.error(f"DagLayer input dimension {x.size(1)} does not match expected {self.total_dim}.")
        
        A_masked = self.A * (1 - torch.eye(self.total_dim, device=self.A.device))
        
        I = torch.eye(self.total_dim, device=self.A.device)
        
        try:
            system_matrix = I - A_masked
            system_matrix_stable = system_matrix + config.eps * I
            
            inverse_system_matrix = torch.inverse(system_matrix_stable)
            output = torch.matmul(x, inverse_system_matrix)
            return output
            
        except RuntimeError as e:
            logger.warning(f"Matrix inversion failed in DAG transform: {e}. Using pseudo-inverse as fallback.")
            try:
                pinv_system_matrix = torch.linalg.pinv(system_matrix)
                return torch.matmul(x, pinv_system_matrix)
            except RuntimeError as e_pinv:
                logger.error(f"Pseudo-inverse also failed in DAG transform: {e_pinv}. Returning original input x.")
                return x

    @property
    def dag_constraint_value(self) -> torch.Tensor:
        return ut.compute_dag_constraint(self.A, self.total_dim)

