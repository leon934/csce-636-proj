import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lp import calculate_m_height_parallel

if __name__ == "__main__":
    m_heights = []
    output_file = "hw-4/HW-4-mHeights"

    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                m_heights = pickle.load(f)
        except EOFError:
            m_heights = []
            
    with open("data/HW-4-n_k_m_P", "rb") as f:
        nkmG = pickle.load(f)

    start_index = len(m_heights)
    total_items = len(nkmG)

    print(f"Resuming from index: {start_index}")

    if start_index < total_items:
        # Loop through the remaining data
        for n, k, m, P in tqdm(nkmG[start_index:], initial=start_index, total=total_items):
            # Form systematic generator matrix G = [I_k | P]
            G = np.hstack((np.eye(k), P))

            result = calculate_m_height_parallel(n, k, m, G)

            m_heights.append(result)

            # Checkpoint the progress
            with open(output_file, "wb") as f:
                pickle.dump(m_heights, f)