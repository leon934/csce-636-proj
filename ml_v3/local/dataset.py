import sqlite3
import math
import numpy as np
import torch
from torch.utils.data import Dataset


class MHeightDataset(Dataset):
    def __init__(self, n, k, m, db_path):
        self.n = n
        self.k = k
        self.m = m

        table_name = f"n-{n}_k-{k}_m-{m}"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        rows = []
        try:
            cursor.execute(f'SELECT p_matrix, m_height FROM "{table_name}"')
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"Error reading from table {table_name}: {e}")
        finally:
            conn.close()

        N = len(rows)
        # Pre-allocate contiguous arrays — __getitem__ becomes a fast index op
        # instead of constructing a new tensor from a Python list each time.
        self.P_data = np.empty((N, k, n - k), dtype=np.float32)
        self.y_data = np.empty((N,),          dtype=np.float32)

        for i, (p_bytes, m_height) in enumerate(rows):
            self.P_data[i] = np.frombuffer(p_bytes, dtype=np.int8).reshape(k, n - k)
            self.y_data[i] = math.log2(max(1.0, float(m_height)))

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        # torch.from_numpy on a pre-allocated array is zero-copy and much faster
        # than torch.tensor() on a Python list element.
        # Normalize to [-1, 1] — values are stored as integers in [-100, 100].
        P = torch.from_numpy(self.P_data[idx].copy()) / 100.0
        y = torch.tensor([self.y_data[idx]], dtype=torch.float32)
        return P, y
