import sys
import logging
from datetime import datetime

sys.path.insert(0, "ml_v3/local")

from train         import train_combo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

combinations = [
    # (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    # (9, 5, 2),
    (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
]

if __name__ == "__main__":
    sync_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n, k, m in combinations:
        train_combo(n, k, m, sync_time)
