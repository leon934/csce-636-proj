from pathlib import Path
import pickle
import sqlite3
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

def update_database(n, k, m):
    table_name = f"n-{n}_k-{k}_m-{m}"

    db_path = Path("data/matrices.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            p_matrix BLOB UNIQUE,
            m_height REAL
        )
    ''')
    conn.commit()

    cursor = conn.execute(f'SELECT p_matrix FROM "{table_name}"')
    existing = {row[0] for row in cursor}

    with open("data/CSCE-636-Project-3-Train-n_k_m_P", "rb") as f:
        nkmPs = pickle.load(f)

    with open("data/CSCE-636-Project-3-Train-mHeights", "rb") as f:
        m_heights = pickle.load(f)

    inserted = 0
    for (curr_n, curr_k, curr_m, P), m_height in zip(nkmPs, m_heights):
        if curr_n == n and curr_k == k and curr_m == m:
            P_bytes = P.astype(np.int8).tobytes()
            if P_bytes not in existing:
                existing.add(P_bytes)
                conn.execute(f'INSERT OR IGNORE INTO "{table_name}" (p_matrix, m_height) VALUES (?, ?)',
                             (P_bytes, m_height))
                inserted += 1

    conn.commit()
    conn.close()
    print(f"[n={n}, k={k}, m={m}] Inserted {inserted} new rows.")

def main():
    args = sys.argv[1:]

    if len(args) != 3:
        print("args: n k m")
        exit(1)

    n, k, m = map(int, args)
    update_database(n, k, m)

if __name__ == "__main__":
    main()
