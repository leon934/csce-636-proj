import subprocess
import sys
import time

def main():
    # The exact 9 parameter combinations you requested
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

    # The name of your worker script
    worker_script = "./data_gen/data_gen.py"
    
    processes = []

    print(f"Starting {len(parameters)} data generation processes...")

    try:
        # Launch all 9 processes concurrently
        for n, k, m in parameters:
            # Command: python data_gen.py <n> <k> <m>
            cmd = [sys.executable, worker_script, str(n), str(k), str(m)]
            
            # Popen runs the command in the background without blocking
            p = subprocess.Popen(cmd)
            processes.append((p, f"n={n}_k={k}_m={m}"))
            
            # Optional: tiny sleep to stagger the exact start times 
            # and ease the initial DB creation/loading barrage
            time.sleep(0.5)

        print("\nAll processes are running. Press Ctrl+C to stop them safely.\n")

        # Keep the main orchestrator alive while children do the work
        for p, name in processes:
            p.wait()

    except KeyboardInterrupt:
        # This is the safety net. If you hit Ctrl+C, it cleans up everything.
        print("\nShutting down safely. Terminating all worker processes...")
        for p, name in processes:
            p.terminate()
            p.wait() # Ensure it actually closed
            print(f"Terminated process for {name}")
        
        print("All processes stopped cleanly.")
        sys.exit(0)

if __name__ == "__main__":
    main()