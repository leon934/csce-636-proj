from pathlib import Path
import pickle
import sqlite3
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lp import calculate_m_height

def gen_valid_P_matrix(n, k, m):
    while True:
        P = np.random.randint(-100, 101, size=(k, n - k), dtype=np.int8)
        G = np.hstack((np.eye(k), P))

        if (m_height := calculate_m_height(n, k, m, G)) != float("inf"):
            return P, m_height

def data_gen(n, k, m):
    table_name = f"n-{n}_k-{k}_m-{m}"
    
    # Ensure directory exists
    db_path = Path("data/matrices.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL") 
    
    # Create table. Quotes are required around table_name because of the hyphens.
    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            p_matrix BLOB UNIQUE,
            m_height REAL
        )
    ''')
    conn.commit()

    # get existing data from og file
    P_matrices = set()

    cursor = conn.execute(f'SELECT p_matrix FROM "{table_name}"')
    for row in cursor:
        P_matrices.add(row[0])

    with open("data/CSCE-636-Project-1-Train-n_k_m_P", "rb") as f:
        orig_nkmPs = pickle.load(f)
    
    with open("data/CSCE-636-Project-1-Train-mHeights", "rb") as f:
        orig_m_heights = pickle.load(f)

    for curr_n, curr_k, curr_m, P, m_height in zip(orig_nkmPs, orig_m_heights):
        P = P.astype(np.int8)

        if curr_n == n and curr_k == k and curr_m == m:
            P_bytes = P.tobytes()

            if P_bytes not in P_matrices:
                P_matrices.add(P_bytes)
                conn.execute(f'INSERT OR IGNORE INTO "{table_name}" (p_matrix, m_height) VALUES (?, ?)', 
                                 (P_bytes, m_height))
    conn.commit()

    # keep saved file as the np arr, not the bytes vers.
    while True:
        P, m_height = gen_valid_P_matrix(n, k, m)
        P_bytes = P.tobytes()

        if P_bytes not in P_matrices:
            P_matrices.add(P_bytes)

            conn.execute(f'INSERT INTO "{table_name}" (p_matrix, m_height) VALUES (?, ?)', (P_bytes, m_height))
            conn.commit()

def main():
    # code is intended to be ran as a process so we can multiprocess datagen for each of the 9 combos
    args = sys.argv[1:]

    if len(args) != 3:
        print("args: n k m")
        exit(1)

    n, k, m = map(int, args)
    data_gen(n, k, m)

if __name__ == "__main__":
    main()