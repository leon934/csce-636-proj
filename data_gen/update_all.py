import subprocess
import sys
import time

def main():
    parameters = [
        (9, 4, 2),
        (9, 4, 3),
        (9, 4, 4),
        (9, 4, 5),
        (9, 5, 2),
        (9, 5, 3),
        (9, 5, 4),
        (9, 6, 2),
        (9, 6, 3),
    ]

    worker_script = "./data_gen/update_database.py"

    processes = []

    print(f"Importing Project-3 data for {len(parameters)} parameter combinations...")

    for n, k, m in parameters:
        cmd = [sys.executable, worker_script, str(n), str(k), str(m)]
        p = subprocess.Popen(cmd)
        processes.append((p, f"n={n}_k={k}_m={m}"))
        time.sleep(0.5)

    for p, name in processes:
        p.wait()

    print("Done.")

if __name__ == "__main__":
    main()
