import torch

def compute_dag_constraint(A: torch.Tensor, m: int) -> torch.Tensor:
    d = m
    
    A_no_diag = A * (1 - torch.eye(d, device=A.device))
    
    eps = 1e-6
    M = torch.eye(d, device=A.device) + A_no_diag/d + eps * torch.eye(d, device=A.device)
    
    E = torch.matrix_power(M, d)
    h_A = torch.trace(E) - d
    
    if h_A.dim() > 0:
        h_A = h_A.mean()
        
    if h_A < 1e-8:
        h_A = torch.tensor(1e-8, device=A.device)
        
    return h_A 