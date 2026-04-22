import torch
import math
import itertools

def augment_matrix(p_matrix):
    """
    Phase 3: Data Augmentation.
    Applies column permutations and sign flips.
    """
    k, cols = p_matrix.shape
    aug_p = p_matrix.clone()
    
    # 1. Sign-flip augmentation (randomly flip signs of rows and/or columns)
    if torch.rand(1).item() > 0.5:
        row_signs = torch.randint(0, 2, (k, 1)).float() * 2 - 1
        aug_p *= row_signs
    if torch.rand(1).item() > 0.5:
        col_signs = torch.randint(0, 2, (1, cols)).float() * 2 - 1
        aug_p *= col_signs
        
    # 2. Column permutation augmentation
    if torch.rand(1).item() > 0.5:
        perm = torch.randperm(cols)
        aug_p = aug_p[:, perm]
        
    return aug_p

def compute_features(p_matrix, n, k, m):
    """
    Phase 2: Feature Engineering.
    Extracts mathematical hints from P and G = [I | P].
    """
    cols = n - k
    
    # Ensure 2D tensor
    if p_matrix.dim() == 1:
        P = p_matrix.view(k, cols)
    else:
        P = p_matrix

    features = []
    
    # 1. Raw matrix elements (flattened)
    features.append(P.flatten())
    
    # 2. SVD and Condition Numbers
    try:
        svd_vals = torch.linalg.svdvals(P)
        cond_num = svd_vals[0] / (svd_vals[-1] + 1e-8)
        min_svd = svd_vals[-1]
    except torch._C._LinAlgError:
        svd_vals = torch.zeros(min(k, cols), device=P.device)
        cond_num = torch.tensor(0.0, device=P.device)
        min_svd = torch.tensor(0.0, device=P.device)
        
    # Pad SVD to max possible size if needed (min(k, cols))
    features.append(svd_vals)
    features.append(cond_num.view(1))
    features.append(min_svd.view(1))
    
    # 3. Row and Column Norms (L1 and L2)
    features.append(torch.norm(P, p=1, dim=0)) # Col L1
    features.append(torch.norm(P, p=2, dim=0)) # Col L2
    features.append(torch.norm(P, p=1, dim=1)) # Row L1
    features.append(torch.norm(P, p=2, dim=1)) # Row L2
    
    # 4. Gram Matrix Diagonals
    gram_cols = P.T @ P
    gram_rows = P @ P.T
    features.append(torch.diag(gram_cols))
    features.append(torch.diag(gram_rows))
    
    # 5. Global Matrix Statistics (Percentiles & Abs Stats)
    abs_P = torch.abs(P)
    features.append(torch.tensor([
        torch.max(abs_P), torch.mean(abs_P), torch.median(abs_P), torch.std(abs_P)
    ], device=P.device))
    
    # 6. Scalar hints
    features.append(torch.tensor([float(n), float(k), float(m)], device=P.device))
    
    return torch.cat(features).float()

def get_feature_dim(n, k, m):
    """Utility to dynamically determine the input size for the MLP."""
    dummy_p = torch.zeros((k, n - k))
    dummy_feat = compute_features(dummy_p, n, k, m)
    return dummy_feat.shape[0]