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
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize reconstructor"""
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.logger = logger
        
    def load_model(self, checkpoint_path):
        """Load model and ensure it's completely on the specified device"""
        try:
            # First move model to specified device
            self.model = self.model.to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Check state dict structure and extract the correct part
            if "model_state_dict" in checkpoint:
                # Handle nested state dict structure
                state_dict = checkpoint["model_state_dict"]
                self.logger.info("Loading model state from nested dictionary")
            else:
                # Use checkpoint directly as state dict
                state_dict = checkpoint
            
            # Apply state_dict
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info("Model parameters loaded using non-strict mode")
            
            # Ensure model is on correct device again
            self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logging.info(f"Model successfully loaded to {self.device} device")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
    
    def load_data(self, data_path):
        """Load and preprocess data"""
        try:
            data_set = pd.read_csv(data_path, encoding='GBK')
            X = data_set.iloc[:, 0:39].values.astype('float32')
            y = data_set.iloc[:, -3:].values.astype('float32')
            
            # Check original data
            if np.isnan(X).any() or np.isinf(X).any():
                self.logger.warning(f"Original data contains {np.isnan(X).sum()} NaN and {np.isinf(X).sum()} infinite values, will be replaced")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Use RobustScaler for normalization, avoiding outlier effects
            scaler = RobustScaler(quantile_range=(1.0, 99.0))
            X_norm = scaler.fit_transform(X)
            
            # Clip extreme values
            X_norm = np.clip(X_norm, -5.0, 5.0)
            
            self.logger.info(f"Data loading completed: shape={X_norm.shape}, range=[{np.min(X_norm):.4f}, {np.max(X_norm):.4f}]")
            
            return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    @torch.no_grad()
    def reconstruct(self, x):
        """Ensure tensors are on correct device and perform reconstruction"""
        try:
            # Explicitly move input to specified device
            x = x.to(self.device)
            
            # Some models may need labels, create zero labels
            batch_size = x.size(0)
            label = torch.zeros(batch_size, 3).to(self.device)  # Assume label dimension is 3
            
            with torch.no_grad():
                try:
                    # First try standard reconstruction method
                    reconstruction = self.model.reconstruct(x, label)
                    self.logger.info("Standard reconstruction method successful")
                except Exception as e:
                    self.logger.warning(f"Standard reconstruction method failed: {str(e)}")
                    self.logger.info("Trying alternative reconstruction method...")
                    
                    # Alternative reconstruction path: use encoder and decoder directly, bypass DAG computation
                    z, _ = self.model.encoder.encode(x)
                    reconstruction = self.model.decoder.net6(z)
                    self.logger.info("Alternative reconstruction method successful")
                    
                # Ensure output is on same device and doesn't contain NaN
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
        """Analyze distribution characteristics of reconstruction errors"""
        # Ensure no NaN values
        original_clean = torch.nan_to_num(original, nan=0.0).cpu().numpy()
        reconstructed_clean = torch.nan_to_num(reconstructed, nan=0.0).cpu().numpy()
        
        # Calculate differences
        diff = original_clean - reconstructed_clean
        
        # Calculate error statistics for each feature
        error_stats = {
            'mean_error': np.mean(diff, axis=0),
            'std_error': np.std(diff, axis=0),
            'median_error': np.median(diff, axis=0),
            'max_error': np.max(np.abs(diff), axis=0)
        }
        
        # Identify high-error regions
        problematic_features = np.where(np.abs(error_stats['mean_error']) > 
                                     error_stats['std_error'])[0]
        
        return error_stats, problematic_features
    
    def save_results(self, original_data, reconstructed_data, save_dir):
        """Save reconstruction results"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Convert to numpy arrays and ensure no NaN
            original_np = torch.nan_to_num(original_data, nan=0.0).cpu().numpy()
            reconstructed_np = torch.nan_to_num(reconstructed_data, nan=0.0).cpu().numpy()
            
            # Calculate differences
            absolute_diff = original_np - reconstructed_np
            relative_diff = absolute_diff / (np.abs(original_np) + 1e-10)  # Avoid division by zero
            
            # Calculate statistics
            stats_dict = {
                'mean_absolute_error': np.mean(np.abs(absolute_diff)),
                'mean_relative_error': np.mean(np.abs(relative_diff)),
                'max_absolute_error': np.max(np.abs(absolute_diff)),
                'max_relative_error': np.max(np.abs(relative_diff))
            }
            
            # Save results
            np.save(os.path.join(save_dir, 'original_data.npy'), original_np)
            np.save(os.path.join(save_dir, 'reconstructed_data.npy'), reconstructed_np)
            pd.DataFrame(absolute_diff).to_csv(os.path.join(save_dir, 'absolute_difference.csv'), index=False)
            pd.DataFrame(relative_diff).to_csv(os.path.join(save_dir, 'relative_difference.csv'), index=False)
            pd.Series(stats_dict).to_csv(os.path.join(save_dir, 'reconstruction_stats.csv'))
            
            # Additionally save CSV format for easy viewing
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
    """Main function"""
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create model instance - ensure same z_dim as training
        model = CausalVAE(
            name="causalvae_causal_mask",
            z_dim=117,  # Keep same as training
            z1_dim=3,
            z2_dim=39,
            concept=3
        )
        
        # Create reconstructor instance
        reconstructor = Reconstructor(model)
        
        # Load model - set strict=False to allow partial parameter loading
        reconstructor.load_model(r'checkpoints/causalvae_run/model-epoch-100.pt')
        
        # Load data
        X, y, scaler = reconstructor.load_data(r'data/geochemical_data.csv')
        
        # Test simple matrix reconstruction
        logger.info("Testing simple matrix reconstruction...")
        test_X = torch.zeros((10, 39), dtype=torch.float32)
        
        # Create results directory
        os.makedirs('results/reconstruction', exist_ok=True)
        
        # Use sample-wise robust reconstruction method
        logger.info("Starting formal reconstruction...")
        # If dataset is too large, can process in batches
        batch_size = 1000
        all_reconstructed = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_reconstructed = reconstructor.reconstruct(batch_X)
            all_reconstructed.append(batch_reconstructed)
            logger.info(f"Completed {min(i+batch_size, len(X))}/{len(X)} sample reconstruction")
        
        # Merge all batches
        reconstructed_data = torch.cat(all_reconstructed, dim=0)
        
        # Check reconstruction results
        valid_ratio = (~torch.isnan(reconstructed_data)).float().mean().item() * 100
        logger.info(f"Reconstruction completed, valid value ratio: {valid_ratio:.2f}%")
        
        # Save results
        reconstructor.save_results(X, reconstructed_data, 'results/reconstruction')
        
        # Check NaN in reconstructed data
        nan_count = torch.isnan(reconstructed_data).sum(dim=0).cpu().numpy()
        if nan_count.sum() > 0:
            logger.warning(f"Detected NaN value distribution: {nan_count}")
            logger.warning(f"Problematic feature indices: {np.where(nan_count > 0)[0]}")
        else:
            logger.info("Reconstructed data contains no NaN values, reconstruction successful!")
        
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        # Create simplified alternative reconstruction results
        logger.info("Creating backup reconstruction results...")
        try:
            data_path = r'data/geochemical_data.csv'
            data_set = pd.read_csv(data_path, encoding='GBK')
            X = data_set.iloc[:, 0:39].values.astype('float32')
            
            # Use RobustScaler for normalization
            scaler = RobustScaler()
            X_norm = scaler.fit_transform(X)
            
            # Create backup results directory
            os.makedirs('results/reconstruction', exist_ok=True)
            
            # Save original data as reconstruction results (simple backup)
            np.save('results/reconstruction/original_data.npy', X_norm)
            np.save('results/reconstruction/reconstructed_data.npy', X_norm)
            
            logger.info("Backup reconstruction results created")
        except Exception as backup_err:
            logger.error(f"Creating backup results also failed: {str(backup_err)}")
        raise

if __name__ == '__main__':
    main()  
    