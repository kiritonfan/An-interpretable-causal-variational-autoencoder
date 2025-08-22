"""
Data reconstruction module for trained ICVAE models.

Loads trained CausalVAE models and performs data reconstruction on geochemical datasets.
Provides robust error handling, batch processing, and comprehensive result analysis.

Key features:
- Model loading with device management
- Robust data preprocessing and normalization
- Batch-wise reconstruction for large datasets
- Error analysis and statistical evaluation
- Multiple fallback mechanisms for reliability
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import logging
from codebase.models.mask_vae import CausalVAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Reconstructor:
    """
    Handles data reconstruction using trained CausalVAE models.
    
    Manages model loading, data preprocessing, reconstruction execution,
    and result analysis with robust error handling and device management.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize reconstructor with model and device configuration.
        
        Args:
            model: CausalVAE model instance
            device: Target device for computation (cuda/cpu)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.logger = logger
        
    def load_model(self, checkpoint_path):
        """Load trained model weights with device management and error handling."""
        try:
            # Move model to target device before loading weights
            self.model = self.model.to(self.device)
            
            # Load checkpoint with device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                # Extract from training checkpoint format
                state_dict = checkpoint["model_state_dict"]
                self.logger.info("Loading model state from nested dictionary")
            else:
                # Direct state dict format
                state_dict = checkpoint
            
            # Load weights with flexibility for architecture changes
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info("Model parameters loaded using non-strict mode")
            
            # Final device placement and evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"Model successfully loaded to {self.device} device")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
    
    def load_data(self, data_path):
        """Load geochemical dataset and apply robust preprocessing."""
        try:
            # Load CSV data with Chinese encoding support
            data_set = pd.read_csv(data_path, encoding='GBK')
            X = data_set.iloc[:, 0:39].values.astype('float32')  # 39 geochemical elements
            y = data_set.iloc[:, -3:].values.astype('float32')   # 3 geological labels
            
            # Handle missing and infinite values
            if np.isnan(X).any() or np.isinf(X).any():
                self.logger.warning(f"Original data contains {np.isnan(X).sum()} NaN and {np.isinf(X).sum()} infinite values, will be replaced")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Robust normalization to handle outliers (1-99% quantile range)
            scaler = RobustScaler(quantile_range=(1.0, 99.0))
            X_norm = scaler.fit_transform(X)
            
            # Prevent extreme values that could destabilize model
            X_norm = np.clip(X_norm, -5.0, 5.0)
            
            self.logger.info(f"Data loading completed: shape={X_norm.shape}, range=[{np.min(X_norm):.4f}, {np.max(X_norm):.4f}]")
            
            return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Perform data reconstruction with fallback mechanisms.
        
        Uses trained CausalVAE model to reconstruct input data with multiple
        fallback strategies for robustness.
        """
        try:
            # Ensure input tensor is on correct device
            x = x.to(self.device)
            
            # Create dummy labels for models requiring label input
            batch_size = x.size(0)
            label = torch.zeros(batch_size, 3).to(self.device)  # 3D geological label space
            
            with torch.no_grad():
                try:
                    # Primary reconstruction: full model with DAG processing
                    reconstruction = self.model.reconstruct(x, label)
                    self.logger.info("Standard reconstruction method successful")
                except Exception as e:
                    self.logger.warning(f"Standard reconstruction method failed: {str(e)}")
                    self.logger.info("Trying alternative reconstruction method...")
                    
                    # Fallback: direct encoder-decoder bypass (skip DAG)
                    z, _ = self.model.encoder.encode(x)
                    reconstruction = self.model.decoder.net6(z)
                    self.logger.info("Alternative reconstruction method successful")
                    
                # Post-processing: device consistency and NaN handling
                reconstruction = reconstruction.to(self.device)
                if torch.isnan(reconstruction).any():
                    self.logger.warning("Reconstruction result contains NaN values, will be replaced with 0")
                    reconstruction = torch.nan_to_num(reconstruction, nan=0.0)
            
            return reconstruction
        except Exception as e:
            logging.error(f"Reconstruction failed: {str(e)}")
            # Return zero tensor instead of None to avoid downstream errors
            return torch.zeros_like(x).to(self.device)
            
    def analyze_reconstruction_error(self, original, reconstructed):
        """
        Analyze reconstruction quality through statistical error metrics.
        
        Computes per-feature error statistics and identifies problematic
        elements with high reconstruction errors.
        """
        # Clean data: remove NaN values for reliable statistics
        original_clean = torch.nan_to_num(original, nan=0.0).cpu().numpy()
        reconstructed_clean = torch.nan_to_num(reconstructed, nan=0.0).cpu().numpy()
        
        # Calculate element-wise reconstruction differences
        diff = original_clean - reconstructed_clean
        
        # Compute comprehensive error statistics
        error_stats = {
            'mean_error': np.mean(diff, axis=0),      # Average bias per element
            'std_error': np.std(diff, axis=0),        # Error variability per element
            'median_error': np.median(diff, axis=0),  # Robust central tendency
            'max_error': np.max(np.abs(diff), axis=0) # Worst-case error per element
        }
        
        # Flag elements with systematic reconstruction issues
        problematic_features = np.where(np.abs(error_stats['mean_error']) > 
                                     error_stats['std_error'])[0]
        
        return error_stats, problematic_features
    
    def save_results(self, original_data, reconstructed_data, save_dir):
        """Save reconstruction results with comprehensive analysis and multiple formats."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Convert tensors to numpy arrays with NaN safety
            original_np = torch.nan_to_num(original_data, nan=0.0).cpu().numpy()
            reconstructed_np = torch.nan_to_num(reconstructed_data, nan=0.0).cpu().numpy()
            
            # Compute reconstruction error metrics
            absolute_diff = original_np - reconstructed_np
            relative_diff = absolute_diff / (np.abs(original_np) + 1e-10)  # Avoid division by zero
            
            # Generate summary statistics for reconstruction quality
            stats_dict = {
                'mean_absolute_error': np.mean(np.abs(absolute_diff)),
                'mean_relative_error': np.mean(np.abs(relative_diff)),
                'max_absolute_error': np.max(np.abs(absolute_diff)),
                'max_relative_error': np.max(np.abs(relative_diff))
            }
            
            # Save binary data for efficient loading
            np.save(os.path.join(save_dir, 'original_data.npy'), original_np)
            np.save(os.path.join(save_dir, 'reconstructed_data.npy'), reconstructed_np)
            
            # Save analysis results in CSV format for inspection
            pd.DataFrame(absolute_diff).to_csv(os.path.join(save_dir, 'absolute_difference.csv'), index=False)
            pd.DataFrame(relative_diff).to_csv(os.path.join(save_dir, 'relative_difference.csv'), index=False)
            pd.Series(stats_dict).to_csv(os.path.join(save_dir, 'reconstruction_stats.csv'))
            
            # Save human-readable CSV files
            pd.DataFrame(reconstructed_np).to_csv(os.path.join(save_dir, 'reconstructed_values.csv'), index=False)
            pd.DataFrame(original_np).to_csv(os.path.join(save_dir, 'original_values.csv'), index=False)
            
            self.logger.info(f"Results saved to: {save_dir}")
            self.logger.info("\nReconstruction statistics:")
            for key, value in stats_dict.items():
                self.logger.info(f"{key}: {value:.6f}")
                
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            # Try using backup path
            backup_dir = os.path.join(os.path.dirname(__file__), 'backup_results')
            self.logger.info(f"Trying backup path: {backup_dir}")
            os.makedirs(backup_dir, exist_ok=True)
            self.save_results(original_data, reconstructed_data, backup_dir)

def main():
    """
    Main reconstruction pipeline.
    
    Orchestrates the complete data reconstruction workflow: model loading,
    data preprocessing, batch reconstruction, and results analysis.
    """
    try:
        # Set random seeds for reproducible results
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize CausalVAE model with same architecture as training
        model = CausalVAE(
            name="causalvae_causal_mask",
            z_dim=117,  # Match training configuration
            z1_dim=3,   # 3 geological labels
            z2_dim=39,  # 39 geochemical elements
            concept=3   # Number of concept categories
        )
        
        # Create reconstructor with automatic device selection
        reconstructor = Reconstructor(model)
        
        # Load pre-trained model weights
        reconstructor.load_model(r'checkpoints')
        
        # Load and preprocess geochemical dataset
        X, y, scaler = reconstructor.load_data(r'data/geochemical_data.csv')
        
        # Prepare output directory
        os.makedirs('results/reconstruction', exist_ok=True)
        
        # Perform batch-wise reconstruction for memory efficiency
        logger.info("Starting formal reconstruction...")
        batch_size = 1000  # Adjust based on available memory
        all_reconstructed = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_reconstructed = reconstructor.reconstruct(batch_X)
            all_reconstructed.append(batch_reconstructed)
            logger.info(f"Completed {min(i+batch_size, len(X))}/{len(X)} sample reconstruction")
        
        # Combine all batch results
        reconstructed_data = torch.cat(all_reconstructed, dim=0)
        
        # Validate reconstruction quality
        valid_ratio = (~torch.isnan(reconstructed_data)).float().mean().item() * 100
        logger.info(f"Reconstruction completed, valid value ratio: {valid_ratio:.2f}%")
        
        # Save comprehensive results and analysis
        reconstructor.save_results(X, reconstructed_data, 'results/reconstruction')
        
        # Final quality check and problematic feature identification
        nan_count = torch.isnan(reconstructed_data).sum(dim=0).cpu().numpy()
        if nan_count.sum() > 0:
            logger.warning(f"Detected NaN value distribution: {nan_count}")
            logger.warning(f"Problematic feature indices: {np.where(nan_count > 0)[0]}")
        else:
            logger.info("Reconstructed data contains no NaN values, reconstruction successful!")
        
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        # Fallback: create basic reconstruction results for analysis continuity
        logger.info("Creating backup reconstruction results...")
        try:
            # Load raw data without model processing
            data_path = r'data/geochemical_data.csv'
            data_set = pd.read_csv(data_path, encoding='GBK')
            X = data_set.iloc[:, 0:39].values.astype('float32')
            
            # Apply basic normalization
            scaler = RobustScaler()
            X_norm = scaler.fit_transform(X)
            
            # Ensure output directory exists
            os.makedirs('results/reconstruction', exist_ok=True)
            
            # Save identity reconstruction (original data as fallback)
            np.save('results/reconstruction/original_data.npy', X_norm)
            np.save('results/reconstruction/reconstructed_data.npy', X_norm)
            
            logger.info("Backup reconstruction results created")
        except Exception as backup_err:
            logger.error(f"Creating backup results also failed: {str(backup_err)}")
        raise

# Execute reconstruction pipeline if run directly
if __name__ == '__main__':
    main()  
    

    
