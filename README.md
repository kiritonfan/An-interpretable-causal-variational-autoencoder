# An-interpretable-causal-variational-autoencoder
This code implements the Interpretable Causal Variational Autoencoder (ICVAE) framework that combines variational autoencoders with causal structure learning to discover causal relationships in geochemical data. The model integrates DAG (Directed Acyclic Graph) constraints with deep generative modeling to learn interpretable causal structures from 39 geochemical elements and 3 geological labels. The framework enables both data reconstruction and causal inference for geological domain applications.

## Environment
This code was developed and tested in the following environment:
**Python**: 3.9  
**PyTorch**: 2.0+ (CUDA 11.8+ recommended for GPU acceleration)  
**Operating System**: Windows 10/11, Linux, macOS

## Required Dependencies
To ensure the code runs properly, please install the following essential libraries:
```bash
torch torchvision numpy pandas scikit-learn matplotlib seaborn networkx argparse logging
```
Detailed dependency installation:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn networkx
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

