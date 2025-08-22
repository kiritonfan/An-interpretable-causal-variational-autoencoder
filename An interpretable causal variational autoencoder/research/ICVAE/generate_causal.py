"""
Causal analysis generator for geochemical data.

Extracts learned causal relationships from trained ICVAE model and generates:
- Detailed causal effect analysis report
- Heatmap visualization of causal influences
- Statistical summary of geological label → element relationships
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from codebase.models.mask_vae import CausalVAE

def generate_complete_causal_analysis():
    """
    Generate complete causal analysis from trained ICVAE model.
    
    Loads trained model and extracts DAG adjacency matrix to analyze
    causal relationships between geological labels and geochemical elements.
    Outputs text report and heatmap visualization.
    """
    
    print("Starting complete causal analysis generation...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load element and label names from dataset
    try:
        data_set = pd.read_csv(os.path.join(project_root, 'data'), encoding='GBK')
        element_names = data_set.columns[:39].tolist()
        label_names = data_set.columns[-3:].tolist()
        
        print(f"Successfully loaded element names (count: {len(element_names)}): {element_names[:5]}...")
        print(f"Successfully loaded label names (count: {len(label_names)}): {label_names}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        element_names = [f"Element_{i+1}" for i in range(39)]
        label_names = ["Fault", "Granite", "Ore-controlling strata"]
        print("Using default element and label names.")
    
    # Load trained ICVAE model
    model_path = os.path.join(project_root, 'checkpoints/causalvae_geo/model-best.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Initialize model with same architecture as training
    model = CausalVAE(
        nn_type='mask',
        z_dim=39,           # VAE latent dimension
        z1_dim=3,           # Number of geological labels  
        z2_dim=39,          # Number of geochemical elements
        concept=3,          # Number of concept categories
        element_relations=True,
        initial=True
    ).to(device)
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model. Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    model.eval()
    
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract learned causal structure from DAG adjacency matrix
    try:
        A = model.dag.A.cpu().detach().numpy()
        
        # Extract label→element causal effects from adjacency matrix
        label_count = len(label_names)
        element_count = len(element_names)
        causal_effects = A[:label_count, label_count:label_count+element_count]
        
        # Analyze and categorize causal relationships
        causal_relations = []
        
        for i in range(label_count):
            for j in range(element_count):
                effect_size = causal_effects[i, j]
                
                # Categorize effect strength
                if effect_size > 0.1:
                    effect_type = "Strong positive"
                elif effect_size < -0.1:
                    effect_type = "Strong negative"
                elif effect_size > 0.03:
                    effect_type = "Weak positive"
                elif effect_size < -0.03:
                    effect_type = "Weak negative"
                elif effect_size > 0:
                    effect_type = "Very weak positive"
                else:
                    effect_type = "Very weak negative"
                
                causal_relations.append((
                    label_names[i],
                    element_names[j],
                    effect_size,
                    effect_type
                ))
        
        # Sort by absolute effect strength
        causal_relations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate detailed causal analysis report
        with open(os.path.join(results_dir, 'complete_causal_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("========== Complete Causal Effect Analysis between Geological Structures and Element Enrichment ==========\n\n")
            f.write("Notes:\n")
            f.write("1. Causal effect size (β) represents the direct causal influence of a cause variable on an effect variable.\n")
            f.write("2. β > 0 indicates a positive effect, β < 0 indicates a negative effect.\n")
            f.write("3. |β| > 0.1 is a strong effect, 0.03 < |β| <= 0.1 is a weak effect, and |β| < 0.03 is a very weak effect.\n\n")
            
            f.write("=== Causal Relationships Grouped by Label ===\n\n")
            
            for label in label_names:
                f.write(f"## Causal Effects of {label} on Elements ##\n\n")
                label_relations = sorted(
                    [rel for rel in causal_relations if rel[0] == label],
                    key=lambda x: abs(x[2]),
                    reverse=True
                )
                
                for i, (cause, effect, beta, effect_type) in enumerate(label_relations):
                    f.write(f"{i+1}. {effect}:\n")
                    f.write(f"   Causal Effect Size (β): {beta:.4f}\n")
                    f.write(f"   Effect Type: {effect_type}\n")
                    f.write("-" * 30 + "\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
            
            f.write("=== All Causal Relationships Sorted by Effect Strength ===\n\n")
            
            for i, (cause, effect, beta, effect_type) in enumerate(causal_relations):
                f.write(f"{i+1}. Causal Path: {cause} → {effect}\n")
                f.write(f"   Causal Effect Size (β): {beta:.4f}\n")
                f.write(f"   Effect Type: {effect_type}\n")
                f.write("-" * 50 + "\n")
        
        print(f"Successfully generated complete causal analysis file: {os.path.join(results_dir, 'complete_causal_analysis.txt')}")
        print(f"Analyzed {len(causal_relations)} causal relationships.")
        
        # Generate causal influence heatmap visualization
        plt.figure(figsize=(20, 8))
        
        causal_matrix = causal_effects.copy()
        
        # Create heatmap with diverging colormap centered at zero
        sns.heatmap(causal_matrix, 
                  cmap='RdBu_r',          # Red-Blue diverging colormap
                  center=0,               # Center colormap at zero
                  annot=True,             # Show numerical values
                  fmt='.2f',              # Format numbers to 2 decimals
                  linewidths=.5,
                  xticklabels=element_names,
                  yticklabels=label_names,
                  cbar_kws={'label': 'Causal Influence Strength'})
        
        plt.title('Complete Causal Influence of Geological Labels on Elements', fontsize=16)
        plt.xlabel('Geochemical Element', fontsize=14)
        plt.ylabel('Geological Label', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        
        plt.gcf().axes[-1].set_ylabel('Causal Influence Strength', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'complete_causal_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully generated complete causal heatmap: {os.path.join(results_dir, 'complete_causal_heatmap.png')}")
        
    except Exception as e:
        print(f"Error during causal analysis generation: {str(e)}")
        import traceback
        traceback.print_exc()

# Run causal analysis if executed directly
if __name__ == "__main__":
    generate_complete_causal_analysis() 

