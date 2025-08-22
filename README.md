# An-interpretable-causal-variational-autoencoder
This code implements the Interpretable Causal Variational Autoencoder (ICVAE) framework by integrating DAG (Directed Acyclic Graph) constraints into variational autoencoders to discover causal relationships in geochemical data. ICVAE is used as the main training and analysis module to learn causal structures from geological data. The reconstruction module handles data reconstruction using trained models for validation. The utils module provides mathematical operations for VAE computations and DAG constraint enforcement, while the CausalVAE model defines the core neural architecture and the neural network module implements encoder-decoder architectures.

## Environment
This code was developed and tested in the following environment:
**Python**: 3.9  
**PyTorch**: 2.0+ (CUDA 11.8+ recommended for GPU acceleration)  


## File Structure & Functions
```
research/ICVAE/
├── ICVAE.py                    #  MAIN SCRIPT - Training, analysis, visualization
├── reconstruct_data.py         #  Data reconstruction using trained models  
├── README.md                   #  This documentation file
├── codebase/                   #  Core model implementation
│   ├── utils.py                #  Math utilities: VAE operations, DAG constraints
│   └── models/
│       ├── mask_vae.py         #  CausalVAE model: encoder+decoder+DAG layer
│       └── nns/
│           └── mask.py         #  Neural networks: Encoder/Decoder architectures
```
        
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

