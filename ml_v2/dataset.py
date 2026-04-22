import sqlite3
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from features import compute_features, augment_matrix, get_feature_dim

class MHeightDataset(Dataset):
    def __init__(self, n, k, m, db_path, is_training=False):
        self.n = n
        self.k = k
        self.m = m
        self.is_training = is_training
        self.data = []
        
        table_name = f"n-{n}_k-{k}_m-{m}"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f'SELECT p_matrix, m_height FROM "{table_name}"')
            rows = cursor.fetchall()
            
            for p_bytes, m_height in rows:
                p_array = np.frombuffer(p_bytes, dtype=np.int8).astype(np.float32)
                p_matrix = p_array.reshape(self.k, self.n - self.k)
                # Phase 1: Transform target to log2 space immediately
                log_target = math.log2(max(1.0, float(m_height)))
                self.data.append((p_matrix, log_target))
                
        except sqlite3.OperationalError as e:
            print(f"Error reading from table {table_name}: {e}")
        finally:
            conn.close()

        # Phase 1: Calculate Standardization (Zero-mean, Unit-variance) stats over the raw features
        self.feature_dim = get_feature_dim(n, k, m)
        if len(self.data) > 0:
            print(f"Calculating normalization statistics for n={n}, k={k}, m={m}...")
            all_feats = [compute_features(torch.tensor(p), n, k, m) for p, _ in self.data]
            all_feats_tensor = torch.stack(all_feats)
            self.mean = all_feats_tensor.mean(dim=0)
            self.std = all_feats_tensor.std(dim=0) + 1e-8 # Prevent divide by zero
        else:
            self.mean = torch.zeros(self.feature_dim)
            self.std = torch.ones(self.feature_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_matrix_np, log_m_height = self.data[idx]
        p_matrix = torch.tensor(p_matrix_np, dtype=torch.float32)
        
        # Phase 3: Apply augmentations only during training
        if self.is_training:
            p_matrix = augment_matrix(p_matrix)
            
        # Phase 2: Compute math hints
        features = compute_features(p_matrix, self.n, self.k, self.m)
        
        # Phase 1: Standardize inputs
        standardized_features = (features - self.mean) / self.std
        
        y = torch.tensor([log_m_height], dtype=torch.float32)
        return standardized_features, y