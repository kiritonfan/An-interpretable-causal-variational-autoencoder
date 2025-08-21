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
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return torch.device("cuda:0")
    
    max_memory = 0
    best_device_index = 0
    for i in range(device_count):
        memory = torch.cuda.get_device_properties(i).total_memory
        if memory > max_memory:
            max_memory = memory
            best_device_index = i
    return torch.device(f"cuda:{best_device_index}")


class ModelConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.eps = 1e-6

    def set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

config = ModelConfig()
config.set_seed()

class CausalVAE(nn.Module):
    def __init__(self, 
                 nn_type: str = 'mask', 
                 name: str = 'vae',
                 z_dim: int = 117, 
                 z1_dim: int = 3,
                 z2_dim: int = 39,
                 concept: int = 3,
                 element_relations: bool = True, 
                 initial: bool = True,
                 device: Optional[torch.device] = None):

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
        # Initialize all model components (encoder, DAG layer, decoder, prior)
        self._init_components(initial)
        
        self.to(self.device)
        logger.info(f"CausalVAE initialized: z_dim(VAE)={z_dim}, z1_dim(labels)={z1_dim}, z2_dim(elements)={z2_dim}")

    def _validate_dimensions(self):
        if self.z1_dim <= 0 or self.z2_dim <= 0:
            raise ValueError("Label and element dimensions must be positive.")
        if self.z_dim < self.z2_dim:
            logger.warning(f"VAE latent dimension ({self.z_dim}) is less than element dimension ({self.z2_dim}), which might be unintended.")

    def _init_components(self, initial: bool):
        self.encoder = mask.Encoder(z_dim=self.z_dim, num_features=self.z2_dim)
        self.project_z_to_elements = nn.Linear(self.z_dim, self.z2_dim)
        self.decoder = mask.Decoder_DAG(
            z_dim=self.z2_dim,
            concept=self.concept, 
            z1_dim=self.z1_dim,
            z2_dim=self.z2_dim
        )
        self.dag = DagLayer(
            z1_dim=self.z1_dim, 
            z2_dim=self.z2_dim, 
            initial=initial
        )
        self._init_prior_network()
        
        if self.element_relations:
            self._init_element_relations()

    def _init_prior_network(self):
        """Build a small MLP that maps discrete labels to the conditional prior
        parameters (mean and variance) of the VAE latent variable. This enables
        label-conditional generation and tighter ELBO bounds.
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
            nn.Linear(256, 2 * self.z_dim)
        )
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
        x = x.to(self.device)
        label = self._ensure_tensor(label).to(self.device)
        
        q_m_raw, q_v_raw_unactivated = self.encoder.encode(x)
        q_v_raw = F.softplus(q_v_raw_unactivated) + config.eps
        z_raw = ut.sample_gaussian(q_m_raw, q_v_raw)

        element_noise = self.project_z_to_elements(z_raw)
        dag_exogenous_input = torch.cat((label, element_noise), dim=1)

        z_structured = self.dag(dag_exogenous_input)
        
        with torch.no_grad():
            A = self.dag.A
            is_root_node_mask = torch.sum(torch.abs(A), dim=1) == 0
        z_for_decoder_input = torch.where(is_root_node_mask, dag_exogenous_input, z_structured)

        z_elements_for_decoding = z_for_decoder_input[:, self.z1_dim:]
        
        x_recon = self.decoder.decode(z_elements_for_decoding)
        
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
        """Compute the negative ELBO and its components.

        Returns a tuple of (NELBO, KL, Reconstruction). All values are scalars.
        """
        try:
            outputs = self.forward(x, label)
            x_hat = outputs['x_recon']
            q_m_raw, q_v_raw = outputs['q_m_raw'], outputs['q_v_raw']
            
            p_m_raw, p_v_raw = (outputs['p_m_raw'], outputs['p_v_raw']) if use_conditional_prior else \
                               (torch.zeros_like(q_m_raw), torch.ones_like(q_v_raw))
            
            kl_raw = ut.kl_normal(q_m_raw, q_v_raw, p_m_raw, p_v_raw)
            kl_raw = self._ensure_scalar(kl_raw.sum(dim=-1))

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
        """Total training loss combining reconstruction, KL, and DAG penalties."""
        _, kl_raw, rec = self.negative_elbo_bound(x, label, use_conditional_prior=use_conditional_prior)
        
        dag_loss_structural, sparsity_loss_A = self._compute_dag_losses()
        
        dag_weight = max(0.01, dag_weight)
        
        total_loss = (rec_weight * rec + 
                      kl_weight * kl_raw + 
                      dag_weight * dag_loss_structural + 
                      sparsity_loss_A)
        
        if torch.isnan(total_loss):
            logger.warning("NaN loss detected, replacing with a safe value.")
            total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        summaries = self._create_loss_summaries(
            total_loss, rec, kl_raw, dag_loss_structural, sparsity_loss_A, 
            rec_weight, kl_weight, dag_weight
        )
        
        return total_loss, summaries

    def _compute_dag_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute acyclicity and sparsity losses for the DAG adjacency matrix."""
        dag_param = self.dag.A
        
        h_a = self._h_A_robust(dag_param, dag_param.shape[0])
        h_a = torch.clamp(h_a, min=config.eps)
        dag_loss_structural = h_a + 0.5 * h_a * h_a
        
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
    def __init__(self, z1_dim: int = 3, z2_dim: int = 39, initial: bool = True):
        super().__init__()
        
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.total_dim = z1_dim + z2_dim
        self.device = _get_optimal_device()

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

