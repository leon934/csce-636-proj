import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset

class MHeightDataset(Dataset):
    def __init__(self, n, k, m, db_path):
        self.n = n
        self.k = k
        self.m = m
        self.data = []
        
        table_name = f"n-{n}_k-{k}_m-{m}"
        
        # Connect to SQLite and fetch data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f'SELECT p_matrix, m_height FROM "{table_name}"')
            rows = cursor.fetchall()
            
            for p_bytes, m_height in rows:
                # Reconstruct the numpy array. Ensure the dtype matches how it was saved.
                p_array = np.frombuffer(p_bytes, dtype=np.int8)
                
                # Reshape to the expected k x (n-k) dimensions
                p_matrix = p_array.reshape(self.k, self.n - self.k)
                
                self.data.append((p_matrix, m_height))
                
        except sqlite3.OperationalError as e:
            print(f"Error reading from table {table_name}: {e}")
        finally:
            conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_matrix, m_height = self.data[idx]
        # Flatten the matrix for the feed-forward network
        x = torch.tensor(p_matrix.flatten(), dtype=torch.float32)
        y = torch.tensor([m_height], dtype=torch.float32)
        return x, y
    
class CombinedMHeightDataset(Dataset):
    """
    Loads all m-height tables for a fixed n and k into a single dataset.
    Returns: (P_matrix, m_val, m_height)
    """
    def __init__(self, n, k, db_path):
        self.n = n
        self.k = k
        self.data = []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Valid m values are from 2 up to n-k
        valid_m_values = range(2, (n - k) + 1)
        
        for m in valid_m_values:
            table_name = f"n-{n}_k-{k}_m-{m}"
            try:
                cursor.execute(f'SELECT p_matrix, m_height FROM "{table_name}"')
                rows = cursor.fetchall()
                for p_bytes, m_height in rows:
                    p_array = np.frombuffer(p_bytes, dtype=np.float32)
                    p_matrix = p_array.reshape(self.k, self.n - self.k)
                    # Store the matrix, the integer m, and the true height
                    self.data.append((p_matrix, float(m), float(m_height)))
            except sqlite3.OperationalError as e:
                print(f"Error reading from table {table_name}: {e}")
                
        conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_matrix, m_val, m_height = self.data[idx]
        x = torch.tensor(p_matrix.flatten(), dtype=torch.float32)
        m_tensor = torch.tensor([m_val], dtype=torch.float32)
        y = torch.tensor([m_height], dtype=torch.float32)
        return x, m_tensor, y