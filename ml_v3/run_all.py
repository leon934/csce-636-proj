import subprocess
import sys
from datetime import datetime


def main():
    sync_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    combinations = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3),
    ]

    processes = []
    print(f"Starting {len(combinations)} parallel training processes (sync_time={sync_time})...")

    for n, k, m in combinations:
        cmd = [
            sys.executable, "ml_v3/train.py",
            "--n", str(n), "--k", str(k), "--m", str(m),
            "--sync_time", sync_time,
        ]
        p = subprocess.Popen(cmd)
        processes.append((p, f"n={n},k={k},m={m}"))
        print(f"  Launched {processes[-1][1]}")

    for p, name in processes:
        p.wait()

    print("All training processes completed.")


if __name__ == "__main__":
    main()
