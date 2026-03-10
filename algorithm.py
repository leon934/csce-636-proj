import multiprocessing
import numpy as np
from itertools import combinations, product
from tqdm import tqdm
import pulp
from collections import defaultdict
from numpy.random import default_rng
import sys

def log_message(msg, n: int, k: int, m: int):
    with open(f"my_log{n}{k}{m}.txt", "a") as f:
        f.write(msg + "\n")

def LP(a: int, b: int, psi, G: np.array, inv_tau: list, X, Y, i):
    k, _ = G.shape

    # Setting up LP problem.
    u = [pulp.LpVariable(f"u{i}") for i in range(k)]
    lp_problem = pulp.LpProblem("LP", pulp.LpMaximize)
    lp_problem += pulp.lpSum([(psi[0] * G[i, a]) * u[i] for i in range(k)])

    # Constraints for LP.
    for j in X:
        lp_problem += pulp.lpSum((psi[inv_tau[j]] * G[i, j] - psi[0] * G[i, a]) * u[i] for i in range(k)) <= 0
        lp_problem += pulp.lpSum((-psi[inv_tau[j]] * G[i, j] * u[i]) for i in range(k)) <= -1

    lp_problem += pulp.lpSum(G[i, b] * u[i] for i in range(k)) == 1
    
    for j in Y:
        lp_problem += pulp.lpSum(G[i, j] * u[i] for i in range(k)) <= 1
        lp_problem += pulp.lpSum(-G[i, j]  * u[i] for i in range(k)) <= 1

    lp_problem.solve(pulp.HiGHS_CMD(msg=False))

    if lp_problem.status == pulp.LpStatusInfeasible:
        return (0, i)
    if lp_problem.status == pulp.LpStatusUnbounded:
        return (float('inf'), i)
    elif lp_problem.status == pulp.LpStatusOptimal:
        return (lp_problem.objective.value(), i)

def LP_wrapper(args):
    return LP(*args)

def build_tasks(G: np.array, m: int, idx: int):
    k, n = G.shape

    n = set(range(n))

    tasks = []

    # Iterates through each combination.
    for a in n:
        for b in n - {a}:
            for X in list(combinations(n - {a, b}, m - 1)):
                Y = n - set(X) - {a, b}
                tau = [a] + sorted(list(X)) + [b] + sorted(list(Y))
                inv_tau = {val : i for i, val in enumerate(tau)}

                for psi in product([-1, 1], repeat=m):
                    tasks.append((a, b, psi, G, inv_tau, X, Y, idx))
    
    return tasks

def generate_neighbor(G: np.array, T: int, T_init: int):
    k, n = G.shape
    num_entries = k * (n - k)
    num_edits = int(np.ceil(T / T_init * (k * (n - k) / 5)))

    P = G[:, k:]
    I = np.identity(k)  # noqa: E741

    while True:
        G_new = G.copy()
        P_new = P.copy()
        P_flat = P_new.flatten()

        rng = default_rng()
        indices = rng.choice(num_entries, size=num_edits, replace=False)

        max_delta = 25

        for i in indices:
            possible_deltas = list(range(-max_delta, 0)) + list(range(1, max_delta + 1))
            delta = np.random.choice(possible_deltas)

            P_flat[i] = np.clip(P_flat[i] + delta, -100, 100)

        G_new[:, k:] = P_flat.reshape((k, n - k))

        # Validation checks.
        has_identity_col = any(
            any(np.all(G_new[:, j] == I[:, i]) for i in range(k))
            for j in range(k, n)
        )

        has_zero_col = any(np.all(G_new[:, j] == 0) for j in range(k, n))

        if not has_zero_col and not has_identity_col and not np.array_equal(G_new, G):
            return G_new

def obtain_best_neighbor_val(G: np.array, T: int, T_init: int, num_neighbor: int, m: int):
    k, n = G.shape

    total_tasks = []
    G_arr = []
    idx_to_G = {}
    
    for i in range(num_neighbor):
        G_i = generate_neighbor(G, T, T_init)
        idx_to_G[i] = G_i

        G_arr.append(G_i)

        total_tasks.extend(build_tasks(G_i, m, i))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        m_heights = list(tqdm(
                        pool.imap(LP_wrapper, total_tasks, chunksize=1),
                        total=len(total_tasks)
                    ))
        
    mheight_map = defaultdict(lambda: float('-inf'))  # track max for each matrix

    for (subtask_val, matrix_id) in m_heights:
        if subtask_val is not None and subtask_val > mheight_map[matrix_id]:
            mheight_map[matrix_id] = subtask_val
    
    best_id, new_m_height = min(mheight_map.items(), key=lambda x: x[1])
    G_new = idx_to_G[best_id]

    return (G_new, new_m_height)

def simulated_annealing(k: int, n: int, m: int, alpha: float, T: int, T_min: float, num_neighbor: int):
    P = np.random.randint(-40, 41, size=(k, n - k))
    I = np.eye(k)  # noqa: E741

    T_init = T

    GLOBAL_MIN = float("inf")
    GLOBAL_G = np.array(0)

    # Calculates current generator matrix's m-height.
    G_curr = np.hstack((I, P))
    total_init_tasks = build_tasks(G_curr, m, 0)

    print("Calculating the initial matrix's m-height.")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
                            pool.imap(LP_wrapper, total_init_tasks),
                            total=len(total_init_tasks)
                        ))

    curr_m_height = max(val for val, _ in results)

    print(f"Initial m-height is: {curr_m_height}")
    print(f"Initial generator matrix is:\n{G_curr}")

    log_message(f"Job: n = {n}    k = {k}     m = {m}", n, k, m)
    log_message(f"Initial generator matrix is:\n{G_curr}", n, k, m)
    log_message(f"Current m-height: \n{curr_m_height}", n, k, m)

    pbar = tqdm(total=int(np.log(T_min / T) / np.log(alpha)))

    while T > T_min:

        print("\nGenerating and obtaining the best m-height in the neighbors.\n")
        G_new, new_m_height = obtain_best_neighbor_val(G_curr, T, T_init, num_neighbor, m)

        # Replace it using simulated annealing.
        if new_m_height < curr_m_height:
            G_curr = G_new
            curr_m_height = new_m_height

            if GLOBAL_MIN < curr_m_height:
                GLOBAL_MIN = curr_m_height
                GLOBAL_G = G_new

            log_message(f"Job: n = {n}    k = {k}     m = {m}", n, k, m)
            log_message(f"New minimum m_height found:\n{curr_m_height}", n, k, m)
            log_message(f"Current graph: \n{G_curr}", n, k, m)

            print(f"Job: n = {n}    k = {k}     m = {m}", flush=True)
            print(f"New minimum m_height found:\n{curr_m_height}", flush=True)
            print(f"Current graph: \n{G_curr}", flush=True)
        else:
            if new_m_height == curr_m_height:
                log_message(f"G_new matrix:\n{G_new}", n, k, m)
                log_message(f"G_new matrix:\n{G_new}", n, k, m)

                print(f"G_new matrix:\n{G_new}", flush=True)
                print(f"G_old matrix:\n{G_curr}", flush=True)

            log_message(f"Current probability: {np.exp(-(new_m_height - curr_m_height) / T)}", n, k, m)
            log_message(f"Current delta m_h: {-(new_m_height - curr_m_height)}", n, k, m)
            log_message(f"Current temperature: {T}\n", n, k, m)

            print(f"Current probability: {np.exp(-(new_m_height - curr_m_height) / T)}", flush=True)
            print(f"Current delta m_h: {-(new_m_height - curr_m_height)}", flush=True)
            print(f"Current temperature: {T}\n", flush=True)

            if np.random.rand() < np.exp(-(new_m_height - curr_m_height) / T):
                G_curr = G_new
                curr_m_height = new_m_height

        # Update temperature.
        T *= alpha
        pbar.update(1)

    print("\n")
    print(f"The best m-height is: {GLOBAL_MIN}")
    print(f"The best matrix is: \n{GLOBAL_G}")

    return GLOBAL_MIN, GLOBAL_G

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py n k m")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
        m = int(sys.argv[3])
    except ValueError:
        print("All arguments must be integers: n k m")
        sys.exit(1)

    # Hyperparameters.
    T = 10
    T_min = 1e-2
    num_neighbor = 8

    alpha = 0.9

    GLOBAL_MIN, GLOBAL_G = simulated_annealing(k, n, m, alpha, T, T_min, num_neighbor)

    log_message(f"The best m-height is: {GLOBAL_MIN}", n, k, m)
    log_message(f"The best matrix is: \n{GLOBAL_G}", n, k, m)

if __name__ == "__main__":
    main()