import subprocess
import sys
from datetime import datetime

def main():
    sync_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    standard_combinations = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3)
    ]

    processes = []
    
    print(f"Starting {len(standard_combinations)} parallel SEPARATE training processes...")
    for n, k, m in standard_combinations:
        print(f"Launching ensemble process for n={n}, k={k}, m={m}")
        cmd = [
            sys.executable, "train.py", 
            "--n", str(n), 
            "--k", str(k), 
            "--m", str(m), 
            "--sync_time", sync_time
        ]
        p = subprocess.Popen(cmd)
        processes.append(p)

    for p in processes:
        p.wait()

    print("All combination training processes have completed.")

if __name__ == "__main__":
    main()