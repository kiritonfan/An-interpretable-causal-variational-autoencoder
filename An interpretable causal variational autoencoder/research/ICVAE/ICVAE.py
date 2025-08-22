"""
ICVAE - Interpretable Causal Variational Autoencoder
===================================================

Main Modules:
- Data Loading: CSV preprocessing, normalization, outlier handling
- Model Training: Dynamic weight scheduling, early stopping, checkpointing  
- Visualization: Causal graphs, training curves, analysis reports
- Utilities: DataLoader setup, EMA, argument parsing

Usage:
    python ICVAE.py --data_path your_data.csv --epoch_max 200
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils import data
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import sys
from tqdm import tqdm
import torch.amp as amp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from codebase.models.mask_vae import CausalVAE
    from codebase.models import nns
except ImportError as e:
    logging.error(f"Failed to import modules: {e}. Please ensure the project structure is correct and all dependencies are installed.")
    sys.exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================

def get_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ICVAE (Interpretable Causal VAE) model for geochemical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # DATA PARAMETERS
    parser.add_argument('--data_path', type=str, 
                        default="data",
                        help="Path to the training data CSV file (39 elements + 3 labels)")
    
    # TRAINING PARAMETERS  
    parser.add_argument('--epoch_max', type=int, default=250, 
                        help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="Training batch size for mini-batch gradient descent")
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help="Initial learning rate for AdamW optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help="L2 weight decay regularization for AdamW optimizer")
    parser.add_argument('--iter_save', type=int, default=10, 
                        help="Save model checkpoint every n epochs")
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="Number of worker processes for parallel data loading")

    # MODEL ARCHITECTURE PARAMETERS
    parser.add_argument('--z_dim', type=int, default=39, 
                        help="VAE latent space dimension (recommended: same as element count)")
    parser.add_argument('--initial', action='store_true', 
                        help="Whether to randomly initialize DAG adjacency matrix")

    # OUTPUT DIRECTORY PARAMETERS
    parser.add_argument('--results_dir', type=str, default='results', 
                        help="Directory to save training curves and analysis plots")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/causalvae_run', 
                        help="Directory to save model checkpoints and best model")
    
    return parser.parse_args()


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(data_path: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load CSV data (39 elements + 3 labels), handle NaN/outliers, normalize with RobustScaler.
    Returns: X_normalized, y_labels, element_names, label_names
    """
    try:
        # Load CSV data with GBK encoding
        data_set = pd.read_csv(data_path, encoding='GBK')
        
        if verbose:
            logging.info(f"Data loaded: {data_set.shape}")
            logging.info(f"Columns: {data_set.columns.tolist()}")
        
        # Extract column names and data arrays
        element_names = data_set.columns[:39].tolist()
        label_names = data_set.columns[-3:].tolist()
        X = data_set.iloc[:, :39].values.astype('float32')
        y = data_set.iloc[:, -3:].values.astype('float32')
        
        if verbose:
            logging.info(f"Elements: {len(element_names)}, Labels: {len(label_names)}")
            logging.info(f"NaN count: {np.isnan(X).sum()}, Zero count: {(X == 0).sum()}")

        # Handle NaN and infinite values
        if np.isnan(X).any() or np.isinf(X).any():
            logging.warning(f"Replacing NaN/Inf values...")
            X = np.nan_to_num(X, nan=0.0, 
                             posinf=np.nanmax(X[np.isfinite(X)]), 
                             neginf=np.nanmin(X[np.isfinite(X)]))
        
        # RobustScaler normalization and zero preservation
        scaler = RobustScaler(quantile_range=(1, 99))
        X_norm = scaler.fit_transform(X)
        mask_zero = (X == 0)
        X_norm_clean = np.clip(X_norm, -5.0, 5.0)
        X_norm_clean[mask_zero] = 0
        
        if verbose:
            logging.info(f"Normalized range: [{X_norm_clean.min():.3f}, {X_norm_clean.max():.3f}]")
        
        # Normalize labels to [0, 1]
        y = np.clip(y, 0, 1)
        
        return X_norm_clean, y, element_names, label_names
        
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_path}")
        logging.error("Please check the file path and ensure the CSV file exists.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during data loading or preprocessing: {e}")
        logging.error("Please check the CSV format: 39 element columns + 3 label columns")
        raise


# =============================================================================
# VISUALIZATION & ANALYSIS
# =============================================================================

def visualize_causal_graph(model: CausalVAE, save_dir: str, label_names: List[str]):
    """Generate heatmap of causal relationships between geological labels."""
    try:
        # Extract DAG matrix and create heatmap
        A = model.dag.A.cpu().detach().numpy()
        logging.info(f"DAG matrix shape: {A.shape}")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(A, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    xticklabels=label_names, yticklabels=label_names,
                    cbar_kws={'label': 'Causal Strength'})
        
        plt.title('Causal Relationships Between Geological Labels', fontsize=16)
        plt.xlabel('Effect Variable', fontsize=12)
        plt.ylabel('Cause Variable', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'label_causal_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Extract significant relationships
        important_relations = []
        threshold = 0.1
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i != j and abs(A[i, j]) > threshold:
                    direction = "Positive" if A[i, j] > 0 else "Negative"
                    important_relations.append((label_names[i], label_names[j], A[i, j], direction))
        
        important_relations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Log significant relationships
        logging.info("\n--- Significant Causal Relationships ---")
        for i, (cause, effect, strength, direction) in enumerate(important_relations):
            logging.info(f"{i+1}. {cause} → {effect}: {abs(strength):.4f} ({direction})")
        
    except Exception as e:
        logging.error(f"Error generating causal heatmap: {str(e)}")
        plt.close()

def plot_training_curves(loss_history, kl_history, rec_history, save_dir):
    """Generate three-panel training curves: total loss, KL divergence, reconstruction error."""
    try:
        plt.figure(figsize=(15, 5))
        
        # Total Loss
        plt.subplot(1, 3, 1)
        plt.plot(loss_history, 'b-', linewidth=2)
        plt.title('Total Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # KL Divergence
        plt.subplot(1, 3, 2)
        plt.plot(kl_history, 'r-', linewidth=2)
        plt.title('KL Divergence', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.grid(True, alpha=0.3)
        
        # Reconstruction Error
        plt.subplot(1, 3, 3)
        plt.plot(rec_history, 'g-', linewidth=2)
        plt.title('Reconstruction Error', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Rec Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        if loss_history:
            logging.info(f"Final losses - Total:{loss_history[-1]:.4f}, KL:{kl_history[-1]:.4f}, Rec:{rec_history[-1]:.4f}")
        
    except Exception as e:
        logging.error(f"Error plotting training curves: {str(e)}")
        plt.close()

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def create_data_loaders(X_norm: np.ndarray, y: np.ndarray, batch_size: int, num_workers: int) -> data.DataLoader:
    """Create optimized PyTorch DataLoader with performance enhancements."""
    dataset = data.TensorDataset(
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    
    generator = torch.Generator().manual_seed(42)
    train_loader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=generator,
        drop_last=True, persistent_workers=(num_workers > 0)
    )
    
    logging.info(f"DataLoader: {len(dataset)} samples, {len(train_loader)} batches")
    return train_loader

class ExponentialMovingAverage:
    """EMA for model parameters. Formula: shadow = decay * shadow + (1 - decay) * current"""
    
    def __init__(self, parameters, decay: float = 0.995):
        """Initialize EMA with model parameters and decay rate."""
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]
        self.collected_params = []

    def update(self):
        """Update EMA parameters after training step."""
        for s_param, param in zip(self.shadow_params, self.parameters):
            s_param.sub_((1 - self.decay) * (s_param - param))

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(model: CausalVAE, train_loader: data.DataLoader, optimizer: optim.Optimizer, 
                args: argparse.Namespace, element_names: List[str], label_names: List[str]):
    """
    Main training loop with dynamic weight scheduling, early stopping, and robustness features.
    Features: KL/DAG weight annealing, EMA, gradient clipping, auto-recovery, adaptive adjustments.
    """
    
    # Initialize training parameters
    min_delta, patience = 0.005, 10
    best_loss, patience_counter, unstable_count = float('inf'), 0, 0
    last_losses = []

    # Dynamic weight annealing schedule
    KL_WEIGHT_MIN, KL_WEIGHT_MAX = 0.5, 1.0
    KL_WARMUP_EPOCHS_START, KL_WARMUP_EPOCHS_DURATION = 10, 20
    DAG_WEIGHT_START, DAG_WEIGHT_MAX = 0.1, 0.5
    DAG_WARMUP_EPOCHS_START, DAG_WARMUP_EPOCHS_DURATION = 15, 15
    
    # Setup training utilities
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_history, kl_history, rec_history = [], [], []
    
    logging.info(f"Training started - KL:[{KL_WEIGHT_MIN}→{KL_WEIGHT_MAX}], DAG:[{DAG_WEIGHT_START}→{DAG_WEIGHT_MAX}]")
    
    # Main training loop
    
    for epoch in range(args.epoch_max):
        
        # Dynamic weight scheduling
        if epoch < KL_WARMUP_EPOCHS_START:
            kl_weight = KL_WEIGHT_MIN
        else:
            progress = (epoch - KL_WARMUP_EPOCHS_START) / KL_WARMUP_EPOCHS_DURATION
            kl_weight = min(KL_WEIGHT_MAX, KL_WEIGHT_MIN + (KL_WEIGHT_MAX - KL_WEIGHT_MIN) * progress)
            
        if epoch < DAG_WARMUP_EPOCHS_START:
            current_dag_weight = DAG_WEIGHT_START
        else:
            progress = (epoch - DAG_WARMUP_EPOCHS_START) / DAG_WARMUP_EPOCHS_DURATION
            current_dag_weight = min(DAG_WEIGHT_MAX, DAG_WEIGHT_START + (DAG_WEIGHT_MAX - DAG_WEIGHT_START) * progress)

        logging.info(f"Epoch {epoch}: KL={kl_weight:.3f}, DAG={current_dag_weight:.3f}")
        
        # Initialize epoch
        model.train()
        epoch_loss = epoch_kl = epoch_rec = epoch_dag = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        batch_count = 0
        
        # Batch training loop
        
        for X_train, y_train in pbar:
            optimizer.zero_grad()
            X_train, y_train = X_train.to(device), y_train.to(device)
            
            try:
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    if torch.isnan(X_train).any() or torch.isnan(y_train).any():
                        logging.warning("NaN in input data, skipping batch")
                        continue
                    
                    # Forward pass with dynamic weights
                    total_loss, summaries = model.loss(
                        x=X_train, label=y_train,
                        rec_weight=1.0, kl_weight=kl_weight, dag_weight=current_dag_weight
                    )

                    rec = summaries['train/rec_loss']
                    kl = summaries['train/kl_raw_loss']
                    dag_loss = summaries['train/dag_structural_loss']
                    
                    if torch.isnan(total_loss):
                        logging.warning("NaN in total loss, skipping batch")
                        continue
                        
                    # Backward pass and optimization
                    total_loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    ema.update()
                    
                    # Accumulate statistics
                    epoch_loss += total_loss.item()
                    epoch_kl += kl
                    epoch_rec += rec
                    epoch_dag += dag_loss
                    batch_count += 1
                    
            except RuntimeError as e:
                logging.warning(f"Runtime error: {e}, skipping batch")
                continue
            
            if batch_count > 0 and batch_count % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'kl': f'{kl:.4f}',
                    'rec': f'{rec:.4f}',
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # ---------------------------------------------------------------------
        # EPOCH COMPLETION AND STATISTICS
        # ---------------------------------------------------------------------
        
        # Validate that at least some batches were processed
        if batch_count == 0:
            logging.error(f"Epoch {epoch}: No valid batches processed! Check data quality.")
            continue

        # Calculate average losses for the epoch
        avg_loss = epoch_loss / batch_count
        avg_kl = epoch_kl / batch_count
        avg_rec = epoch_rec / batch_count
        avg_dag = epoch_dag / batch_count
        
        # Store loss history for visualization
        loss_history.append(avg_loss)
        kl_history.append(avg_kl)
        rec_history.append(avg_rec)
        
        logging.info(f"\nEpoch {epoch} Results: "
                     f"Total Loss={avg_loss:.4f}, KL={avg_kl:.4f}, "
                     f"Reconstruction={avg_rec:.4f}, DAG={avg_dag:.4f}")
        
        # ---------------------------------------------------------------------
        # MODEL VALIDATION AND RECOVERY
        # ---------------------------------------------------------------------
        
        # Check for NaN in model parameters (indicates training instability)
        has_nan = any(torch.isnan(p).any() for p in model.parameters())
        if has_nan:
            logging.error(f"Model parameters contain NaN after Epoch {epoch}! Attempting recovery...")
            try:
                # Restore from best checkpoint if available
                checkpoint_path = os.path.join(args.checkpoint_dir, 'model-best.pt')
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info("Successfully restored model from best checkpoint.")
            except Exception as e:
                logging.critical(f"Model recovery failed: {e}. Training terminated.")
                break
            continue
        
        # ---------------------------------------------------------------------
        # MODEL CHECKPOINTING AND EARLY STOPPING
        # ---------------------------------------------------------------------
        
        # Save best model if significant improvement is achieved
        if not np.isnan(avg_loss) and avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience counter
            save_path = os.path.join(args.checkpoint_dir, 'model-best.pt')
            
            # Prepare EMA state dictionary for saving
            ema_state_dict = {
                k: v.clone() for k, v in zip((p[0] for p in model.named_parameters()), ema.shadow_params)
            }
            
            # Save comprehensive checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'loss_history': loss_history,
                'kl_history': kl_history,
                'rec_history': rec_history
            }, save_path)
            logging.info(f'✓ Saved best model (loss: {best_loss:.4f}) to {save_path}')
        else:
            patience_counter += 1
            logging.info(f"No improvement, patience: {patience_counter}/{patience}")

        # Periodic backup checkpoints
        if epoch > 0 and epoch % args.iter_save == 0:
            backup_path = os.path.join(args.checkpoint_dir, f'model-epoch-{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, backup_path)
            logging.info(f'✓ Saved periodic checkpoint to {backup_path}')

        # Early stopping check
        if patience_counter >= patience:
            logging.warning(f"Early stopping triggered: no improvement for {patience} epochs.")
            break

        # ---------------------------------------------------------------------
        # LEARNING RATE SCHEDULING AND VISUALIZATION
        # ---------------------------------------------------------------------
        
        # Generate training curves periodically for monitoring
        if epoch % 5 == 0 or epoch == args.epoch_max - 1:
            plot_training_curves(loss_history, kl_history, rec_history, args.results_dir)
            
        # Learning rate reduction on plateau
        scheduler.step(avg_loss)
        
        # ---------------------------------------------------------------------
        # ADAPTIVE TRAINING ADJUSTMENTS
        # ---------------------------------------------------------------------
        
        # Adaptive KL weight adjustment based on reconstruction quality
        if epoch > 20 and avg_rec < 1.0:  # If reconstruction is too good
            kl_weight = min(1.5, kl_weight * 1.05)  # Increase KL weight
            if avg_kl < 0.05:  # If KL is too low
                kl_weight = kl_weight * 1.1  # Further increase
        
        # DAG sparsity monitoring and adjustment
        if epoch % 5 == 0:
            dag_matrix = model.dag.A.detach().cpu().numpy()
            non_zero_ratio = (np.abs(dag_matrix) > 1e-4).mean()
            logging.info(f"DAG sparsity: {non_zero_ratio:.4f} non-zero ratio")
            
            # Add exploration noise if DAG becomes too sparse
            if non_zero_ratio < 0.05:
                logging.warning("DAG too sparse, adding exploration noise...")
                with torch.no_grad():
                    noise = 0.05 * torch.randn_like(model.dag.A)
                    mask = torch.rand_like(model.dag.A) > 0.8  # Sparse noise mask
                    model.dag.A.data += noise * mask.float()
                    model.dag.A.data.fill_diagonal_(0)  # Maintain no self-loops
        
        # Training stability monitoring
        last_losses.append(avg_loss)
        if len(last_losses) > 5: 
            last_losses.pop(0)  # Keep only recent losses
            
        # Detect consecutive loss increases (training instability)
        if len(last_losses) >= 3 and avg_loss > last_losses[-2] > last_losses[-3]:
            unstable_count += 1
            logging.warning(f"Training instability: {unstable_count} consecutive loss increases.")
            
            # Emergency learning rate reduction
            if unstable_count >= 3:
                logging.critical("Severe training instability! Reducing learning rate.")
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.1  # Reduce learning rate by 10x
                logging.info(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}")
                unstable_count = 0
        else:
            unstable_count = 0  # Reset if training is stable

    # =========================================================================
    # POST-TRAINING ANALYSIS AND FINAL RESULTS
    # =========================================================================
    
    logging.info("Training completed! Generating comprehensive analysis...")
    
    # Generate final training curves
    plot_training_curves(loss_history, kl_history, rec_history, args.results_dir)
    
    # Load best model for analysis (ensures we analyze the best-performing version)
    best_model_path = os.path.join(args.checkpoint_dir, 'model-best.pt')
    if os.path.exists(best_model_path):
        logging.info(f"Loading best model from {best_model_path} for analysis...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Best model loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")

    # Generate comprehensive causal analysis visualizations
    logging.info("Generating causal structure analysis...")
    
    # 1. Label-to-label relationship heatmap
    visualize_causal_graph(model, args.results_dir, label_names)
    
    # 2. Complete DAG network visualization
    visualize_complete_dag(model, args.results_dir, element_names, label_names)
    
    # 3. Statistical analysis of causal relationships
    analyze_label_element_relationships(model, args.results_dir, element_names, label_names)
    
    # 4. Comprehensive influence heatmap
    visualize_label_element_heatmap(model, args.results_dir, element_names, label_names)
    
    logging.info(f"✓ All analysis results saved to '{args.results_dir}' directory.")
    logging.info("Training and analysis pipeline completed successfully!")
    
    return loss_history, kl_history, rec_history


def visualize_complete_dag(model, save_dir, element_names, label_names):
    """Improved DAG visualization, showing only label-to-element causal relationships."""
    try:
        A = model.get_dag_params().cpu().numpy()
        
        expected_size = len(label_names) + len(element_names)
        if A.shape[0] != expected_size:
            logging.warning(f"⚠️ Warning: DAG matrix size ({A.shape[0]}) does not match label + element count ({expected_size}). Resizing.")
            
            new_A = np.zeros((expected_size, expected_size))
            min_dim = min(A.shape[0], expected_size)
            new_A[:min_dim, :min_dim] = A[:min_dim, :min_dim]
            A = new_A
        
        label_indices = list(range(len(label_names)))
        element_indices = list(range(len(label_names), len(label_names) + len(element_names)))
        
        logging.info("\nSample values from DAG matrix:")
        logging.info("Label→Element submatrix sample:")
        for i in range(min(3, len(label_indices))):
            for j in range(min(5, len(element_indices))):
                logging.info(f"  {label_names[i]}→{element_names[j]}: {A[label_indices[i], element_indices[j]]:.4f}")
        
        G = nx.DiGraph()
        
        all_names = []
        node_colors = []
        node_sizes = []
        
        for i, name in enumerate(label_names):
            G.add_node(i, name=name, type='label')
            all_names.append(name)
            node_colors.append('#ff6666')  # Red for labels
            node_sizes.append(1000)        # Larger nodes for labels
        
        for i, name in enumerate(element_names):
            idx = i + len(label_names)
            G.add_node(idx, name=name, type='element')
            all_names.append(name)
            node_colors.append('#6699cc')  # Blue for elements
            node_sizes.append(700)         # Smaller nodes for elements
        
        non_zero = np.abs(A[A != 0])
        threshold = np.percentile(non_zero, 30) if len(non_zero) > 0 else 0.01
        threshold = max(0.03, threshold)
        logging.info(f"Using threshold {threshold:.4f} to filter weak connections")
        
        # Add only label→element edges
        edges = []
        edge_weights = []
        for i in label_indices:
            for j in element_indices:
                if np.abs(A[i, j]) > threshold:
                    G.add_edge(i, j, weight=np.abs(A[i, j]))
                    edges.append((i, j))
                    edge_weights.append(np.abs(A[i, j]))
        
        logging.info(f"Number of nodes in graph: {G.number_of_nodes()}")
        logging.info(f"Number of edges in graph: {G.number_of_edges()}")
        
        if G.number_of_edges() == 0:
            logging.warning("Warning: No edges found with the current threshold, retrying with a lower threshold...")
            for i in label_indices:
                for j in element_indices:
                    if np.abs(A[i, j]) > 0.01:
                        G.add_edge(i, j, weight=np.abs(A[i, j]))
                        edges.append((i, j))
                        edge_weights.append(np.abs(A[i, j]))
            logging.info(f"Number of edges after using lower threshold: {G.number_of_edges()}")
        
        if len(edges) > 0:
            plt.figure(figsize=(24, 16))
            
            # Grouped layout: labels on the left, elements on the right
            pos_grouped = {}
            
            for i, label in enumerate(label_names):
                pos_grouped[i] = np.array([-10, 20 * (len(label_names) - i - 1) / len(label_names)])
            
            cols = 5
            for i, elem in enumerate(element_names):
                idx = i + len(label_names)
                row = i // cols
                col = i % cols
                pos_grouped[idx] = np.array([col * 4 + 5, -row * 2])
            
            edge_widths = [1.0 + 5.0 * G[u][v]['weight'] for u, v in G.edges()]
            edge_colors = ['#ff9999' for _ in G.edges()]
            
            nx.draw_networkx_nodes(G, pos_grouped, 
                                node_size=node_sizes, 
                                node_color=node_colors, 
                                alpha=0.8)
            
            nx.draw_networkx_edges(G, pos_grouped, 
                                width=edge_widths, 
                                alpha=0.7, 
                                edge_color=edge_colors, 
                                arrowsize=15, 
                                arrowstyle='->', 
                                connectionstyle='arc3,rad=0.2')
            
            nx.draw_networkx_labels(G, pos_grouped, 
                                 {i: G.nodes[i]['name'] for i in G.nodes()}, 
                                 font_size=12, 
                                 font_weight='bold')
            
            from matplotlib.patches import Patch, ConnectionPatch
            legend_elements = [
                Patch(facecolor='#ff6666', edgecolor='black', label='Geological Label'),
                Patch(facecolor='#6699cc', edgecolor='black', label='Geochemical Element'),
                ConnectionPatch((0,0), (1,1), "data", "data", arrowstyle="->", 
                             color='#ff9999', label='Label→Element')
            ]
            plt.legend(handles=legend_elements, fontsize=12)
            
            plt.title('Causal Graph from Labels to Elements', fontsize=20)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'label_to_element_dag.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            sorted_edges = sorted(zip(edges, edge_weights), key=lambda x: x[1], reverse=True)
            logging.info("\nTop 10 Label→Element Causal Relationships:")
            for i, (edge, weight) in enumerate(sorted_edges[:min(10, len(sorted_edges))]):
                source, target = edge
                source_name = G.nodes[source]['name']
                target_name = G.nodes[target]['name']
                logging.info(f"{i+1}. {source_name} → {target_name}: Strength {weight:.4f}")
        
    except Exception as e:
        logging.error(f"Error generating DAG graph: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close()

def analyze_label_element_relationships(model, save_dir, element_names, label_names):
    """Causal effect analysis function."""
    try:
        A = model.dag.A.cpu().detach().numpy()
        
        label_count = len(label_names)
        element_count = len(element_names)
        causal_effects = A[:label_count, label_count:label_count+element_count]
        
        effect_threshold = np.percentile(np.abs(causal_effects[causal_effects != 0]), 50)
        min_threshold = 0.03
        effect_threshold = max(min_threshold, effect_threshold)
        
        causal_relations = []
        
        for i in range(label_count):
            for j in range(element_count):
                effect_size = causal_effects[i, j]
                if np.abs(effect_size) > effect_threshold:
                    if abs(effect_size) > 0.1:
                        effect_type = "Strong positive causal effect" if effect_size > 0 else "Strong negative causal effect"
                    else:
                        effect_type = "Weak positive causal effect" if effect_size > 0 else "Weak negative causal effect"
                    
                    causal_relations.append((
                        label_names[i],
                        element_names[j],
                        effect_size,
                        effect_type
                    ))
        
        causal_relations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        report_path = os.path.join(save_dir, 'causal_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("========== Causal Effect Analysis of Geological Structures on Element Enrichment ==========\n\n")
            f.write("Notes:\n")
            f.write("1. Causal effect size (β) represents the direct causal influence of the cause variable on the effect variable.\n")
            f.write("2. β > 0 indicates a positive causal effect, β < 0 indicates a negative causal effect.\n")
            f.write("3. |β| > 0.1 is a strong causal effect, 0.03 < |β| ≤ 0.1 is a weak causal effect.\n\n")
            
            for i, (cause, effect, beta, effect_type) in enumerate(causal_relations[:117]):
                f.write(f"{i+1}. Causal Path: {cause} → {effect}\n")
                f.write(f"   Causal Effect Size (β): {beta:.4f}\n")
                f.write(f"   Effect Type: {effect_type}\n")
                f.write("-" * 50 + "\n")
        
        logging.info(f"Causal effect analysis report saved to {report_path}")

        if len(causal_relations) > 0:
            plt.figure(figsize=(14, 10))
            
            top_n = min(20, len(causal_relations))
            top_pairs = causal_relations[:top_n]
            causal_paths = [f"{cause} → {effect}" for cause, effect, _, _ in reversed(top_pairs)]
            effect_sizes = [beta for _, _, beta, _ in reversed(top_pairs)]
            
            colors = ['#2166ac' if beta > 0 else '#b2182b' for beta in effect_sizes]
            abs_effect_sizes = [abs(beta) for beta in effect_sizes]

            bars = plt.barh(causal_paths, abs_effect_sizes, color=colors)
            
            for bar, beta in zip(bars, effect_sizes):
                plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                         f"β={beta:.3f}", va='center', ha='left')
            
            plt.xlabel('Causal Effect Size |β|')
            plt.ylabel('Causal Path (Cause → Effect)')
            plt.title(f'Top {top_n} Main Causal Effect Analysis', fontsize=16)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, 'causal_effects_barchart.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Main causal effect bar chart saved to {save_path}")
            
    except Exception as e:
        logging.error(f"Error during causal analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close()

def visualize_label_element_heatmap(model, save_dir, element_names, label_names):
    """Generates a heatmap of causal relationships between labels and elements."""
    try:
        A = model.get_dag_params().cpu().numpy()
        
        label_count = len(label_names)
        element_count = len(element_names)
        
        label_to_element = A[:label_count, label_count:label_count+element_count]
        
        plt.figure(figsize=(20, 8))
        
        sns.heatmap(label_to_element, 
                  cmap='RdBu_r',
                  center=0,
                  annot=False,
                  fmt='.2f',
                  linewidths=.5,
                  xticklabels=element_names,
                  yticklabels=label_names,
                  cbar_kws={'label': 'Causal Influence Strength'})
        
        plt.title('Causal Influence of Geological Labels on Elements', fontsize=16)
        plt.xlabel('Geochemical Element', fontsize=14)
        plt.ylabel('Geological Label', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        
        plt.gcf().axes[-1].set_ylabel('Causal Influence Strength', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'label_to_element_causal.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error generating causal heatmap: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close()

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

if __name__ == '__main__':
    
    # -------------------------------------------------------------------------
    # REPRODUCIBILITY AND ENVIRONMENT SETUP
    # -------------------------------------------------------------------------
    
    # Set random seeds for reproducible results
    torch.manual_seed(42)              # PyTorch random seed
    np.random.seed(42)                 # NumPy random seed
    torch.backends.cudnn.benchmark = True      # Optimize CUDA performance
    torch.backends.cudnn.deterministic = False  # Allow some randomness for speed
    
    # Parse command line arguments
    args = get_args()
    
    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)      # For plots and analysis
    os.makedirs(args.checkpoint_dir, exist_ok=True)   # For model checkpoints
    
    # -------------------------------------------------------------------------
    # DATA LOADING AND PREPROCESSING
    # -------------------------------------------------------------------------
    
    logging.info("="*60)
    logging.info("ICVAE TRAINING PIPELINE STARTED")
    logging.info("="*60)
    
    logging.info("Loading and preprocessing geochemical data...")
    X_norm, y, element_names, label_names = load_data(args.data_path, verbose=True)
    logging.info("✓ Data loading and preprocessing completed.")
    
    # Create optimized data loader for training
    train_loader = create_data_loaders(X_norm, y, args.batch_size, args.num_workers)
    
    # -------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    
    # Extract data dimensions
    num_elements = X_norm.shape[1]  # Should be 39 geochemical elements
    num_labels = y.shape[1]         # Should be 3 geological labels
    
    # Initialize ICVAE model with specified architecture
    model = CausalVAE(
        nn_type='mask',              # Use mask-based neural networks
        z_dim=args.z_dim,           # VAE latent space dimension
        z1_dim=num_labels,          # Geological label dimension (3)
        z2_dim=num_elements,        # Geochemical element dimension (39)
        concept=num_labels,         # Number of geological concepts
        initial=args.initial        # Whether to randomly initialize DAG
    ).to(device)

    # Initialize AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,      # Learning rate
        weight_decay=args.weight_decay,  # L2 regularization
        eps=1e-8                    # Numerical stability
    )
    
    # -------------------------------------------------------------------------
    # TRAINING CONFIGURATION SUMMARY
    # -------------------------------------------------------------------------
    
    logging.info("\n" + "="*50)
    logging.info("TRAINING CONFIGURATION SUMMARY")
    logging.info("="*50)
    logging.info(f"Computing Device: {device}")
    logging.info(f"Model Architecture: CausalVAE")
    logging.info(f"  - VAE Latent Dimension: {args.z_dim}")
    logging.info(f"  - Geological Labels: {num_labels}")
    logging.info(f"  - Geochemical Elements: {num_elements}")
    logging.info(f"Training Parameters:")
    logging.info(f"  - Maximum Epochs: {args.epoch_max}")
    logging.info(f"  - Batch Size: {args.batch_size}")
    logging.info(f"  - Learning Rate: {args.learning_rate}")
    logging.info(f"  - Weight Decay: {args.weight_decay}")
    logging.info(f"Output Directories:")
    logging.info(f"  - Results: {args.results_dir}")
    logging.info(f"  - Checkpoints: {args.checkpoint_dir}")
    logging.info("="*50 + "\n")
    
    # -------------------------------------------------------------------------
    # MODEL TRAINING
    # -------------------------------------------------------------------------
    
    # Execute main training loop
    train_model(model, train_loader, optimizer, args, element_names, label_names)

    logging.info("\n" + "="*60)
    logging.info("ICVAE TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("="*60)
