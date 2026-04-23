import sqlite3
import math
import numpy as np
import torch
from torch.utils.data import Dataset


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
            for p_bytes, m_height in cursor.fetchall():
                p_array = np.frombuffer(p_bytes, dtype=np.int8).astype(np.float32)
                p_matrix = p_array.reshape(k, n - k)
                log_target = math.log2(max(1.0, float(m_height)))
                self.data.append((p_matrix, log_target))
        except sqlite3.OperationalError as e:
            print(f"Error reading from table {table_name}: {e}")
        finally:
            conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_np, log_target = self.data[idx]
        P = torch.tensor(p_np, dtype=torch.float32)  # (k, n-k)

        if self.is_training:
            # Column permutations of P preserve m-height (relabeling parity positions
            # leaves the max-over-all-subsets definition of h_m unchanged).
            perm = torch.randperm(self.n - self.k)
            P = P[:, perm]
            # Negating a column of P preserves m-height (h_m depends only on |c_j|,
            # so scaling any column by -1 leaves every LP objective value unchanged).
            col_signs = torch.randint(0, 2, (1, self.n - self.k)).float() * 2 - 1
            P = P * col_signs

        y = torch.tensor([log_target], dtype=torch.float32)
        return P, y
