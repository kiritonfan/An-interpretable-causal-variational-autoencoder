# An-interpretable-causal-variational-autoencoder
This code implements the Interpretable Causal Variational Autoencoder (ICVAE) framework by integrating DAG (Directed Acyclic Graph) constraints into variational autoencoders to discover causal relationships in geochemical data. ICVAE is used as the main training and analysis module to learn causal structures from geological data. The reconstruction module handles data reconstruction using trained models for validation. The utils module provides mathematical operations for VAE computations and DAG constraint enforcement, while the mask_vae module defines the core CausalVAE model integrating encoder, decoder, DAG layer, and conditional prior networks, and the mask module implements the fundamental neural network architectures including the variational encoder for mapping geochemical elements to latent distributions and the DAG-aware decoder for reconstructing element concentrations from structured latent variables.

## Environment
This code was developed and tested in the following environment:
**Python**: 3.9  
**PyTorch**: 2.4.1 (CUDA 11.8+ recommended for GPU acceleration)  

## Requirements
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- NetworkX
- tqdm

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

## Acknowledgements
This research was supported by the National Key Research and Development Program of China (NO. 22023YFC2906404) , and the Key Research and Development Program of Xinjiang Uygur Autonomous Region, China (2024B03010-3).
        
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

