import subprocess
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Orchestrate training based on architecture type.")
    parser.add_argument("--arch", type=str, default="SimpleMLP", help="Model class in architectures.py")
    args = parser.parse_args()

    arch_name = args.arch
    sync_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the groupings
    theory_models = ["TheoryMLP", "TheoryColumnCNN"]
    standard_combinations = [
        (9, 4, 2), 
        # (9, 4, 3), (9, 4, 4), 
        # (9, 4, 5),
        # (9, 5, 2), (9, 5, 3), (9, 5, 4),
        # (9, 6, 2), (9, 6, 3)
    ]
    # For theory models, we only iterate over n and k (m is handled inside)
    theory_combinations = [
        # (9, 4), 
        (9, 5),
        # (9, 6)
    ]

    processes = []
    
    if arch_name in theory_models:
        print(f"Starting {len(theory_combinations)} parallel THEORY training processes using: {arch_name}...")
        for n, k in theory_combinations:
            print(f"Launching theory process for n={n}, k={k}")
            cmd = [
                sys.executable, "ml/train_same_nk.py", 
                "--n", str(n), 
                "--k", str(k), 
                "--sync_time", sync_time,
                "--arch", arch_name
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
            
    else:
        print(f"Starting {len(standard_combinations)} parallel SEPARATE training processes using: {arch_name}...")
        for n, k, m in standard_combinations:
            print(f"Launching standard process for n={n}, k={k}, m={m}")
            cmd = [
                sys.executable, "ml/train_separate.py", 
                "--n", str(n), 
                "--k", str(k), 
                "--m", str(m), 
                "--sync_time", sync_time,
                "--arch", arch_name
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)

    for p in processes:
        p.wait()

    print(f"All {arch_name} training processes have completed.")

if __name__ == "__main__":
    main()