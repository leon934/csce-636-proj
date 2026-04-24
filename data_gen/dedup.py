"""
Deduplicate matrices.db by collapsing orbits under the two column symmetries
of the m-height function, then write the result to data/matrices_v2.db.

Symmetries (both provably preserve m-height):
  1. Column permutations of P  — relabels parity positions; h_m takes max over
                                  all subsets, so the labeling is irrelevant.
  2. Column sign flips of P    — h_m depends only on |c_j|, so negating any
                                  column of G leaves every LP objective unchanged.

Canonical form for each equivalence class:
  - Sign-normalize each column so its first non-zero element is positive.
  - Sort the sign-normalized columns lexicographically.

The output DB stores one canonical representative per equivalence class.
Training augmentation (column permutations + sign flips) in ml_v3 restores
the full diversity at training time without redundant stored rows.
"""

from pathlib import Path
import sqlite3
import numpy as np

CHUNK = 50_000

COMBINATIONS = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]


def canonical_form(P: np.ndarray) -> np.ndarray:
    """
    Returns the canonical representative of P under column permutations and
    column sign flips. P has shape (k, n-k) and dtype int8.
    """
    cols = [P[:, j].copy() for j in range(P.shape[1])]

    # Sign-normalize: make the first non-zero element of each column positive.
    for col in cols:
        for val in col:
            if val != 0:
                if val < 0:
                    col *= -1
                break

    # Sort columns lexicographically to canonicalize permutations.
    cols.sort(key=lambda c: c.tolist())

    return np.column_stack(cols).astype(np.int8)


def canonical_batch(p_bytes_list: list, k: int, n_minus_k: int) -> list:
    """Compute canonical forms for a batch of raw p_bytes rows."""
    out = []
    for p_bytes in p_bytes_list:
        P = np.frombuffer(p_bytes, dtype=np.int8).reshape(k, n_minus_k)
        out.append(canonical_form(P))
    return out


def dedup_table(src_conn: sqlite3.Connection, dst_conn: sqlite3.Connection,
                n: int, k: int, m: int) -> tuple[int, int]:
    table = f"n-{n}_k-{k}_m-{m}"

    dst_conn.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table}" (
            p_matrix BLOB UNIQUE,
            m_height REAL
        )
    ''')
    dst_conn.commit()

    try:
        total_src = src_conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except sqlite3.OperationalError:
        print(f"  [{table}] not found in source, skipping.")
        return 0, 0

    cursor = src_conn.execute(f'SELECT p_matrix, m_height FROM "{table}"')

    inserted = 0
    read = 0

    while True:
        rows = cursor.fetchmany(CHUNK)
        if not rows:
            break

        p_bytes_list = [r[0] for r in rows]
        m_heights    = [r[1] for r in rows]
        canons       = canonical_batch(p_bytes_list, k, n - k)

        dst_conn.executemany(
            f'INSERT OR IGNORE INTO "{table}" (p_matrix, m_height) VALUES (?, ?)',
            [(c.tobytes(), float(mh)) for c, mh in zip(canons, m_heights)]
        )
        dst_conn.commit()

        read += len(rows)
        print(f"  [{table}] {read:>9,} / {total_src:,} read ...", end="\r", flush=True)

    after = dst_conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    removed = total_src - after
    pct = 100.0 * removed / total_src if total_src else 0.0
    print(f"  [{table}] {total_src:>9,} → {after:,}  ({removed:,} removed, {pct:.1f}% reduction)")
    return total_src, after


def main():
    src_path = Path("data/matrices.db")
    dst_path = Path("data/matrices_v2.db")

    if not src_path.exists():
        print(f"Source DB not found: {src_path}")
        return

    src_conn = sqlite3.connect(src_path, timeout=60.0)
    src_conn.execute("PRAGMA journal_mode=WAL")

    dst_conn = sqlite3.connect(dst_path, timeout=60.0)
    dst_conn.execute("PRAGMA journal_mode=WAL")
    dst_conn.execute("PRAGMA synchronous=NORMAL")

    total_before = 0
    total_after  = 0

    for n, k, m in COMBINATIONS:
        before, after = dedup_table(src_conn, dst_conn, n, k, m)
        total_before += before
        total_after  += after

    src_conn.close()
    dst_conn.close()

    removed = total_before - total_after
    pct = 100.0 * removed / total_before if total_before else 0.0
    print(f"\nDone. Total: {total_before:,} → {total_after:,}  ({removed:,} removed, {pct:.1f}% reduction)")
    print(f"Output written to {dst_path.resolve()}")


if __name__ == "__main__":
    main()
