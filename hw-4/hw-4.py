import pulp
import numpy as np
from itertools import combinations, product
from tqdm import tqdm

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

def calculate_m_height(n, k, m, G):
    """
    Generates all tasks and runs LP sequentially to find the max m-height.
    """
    n_indices = set(range(n))
    max_m_height = float('-inf')

    tasks = []
    
    # 1. Generate all task parameters
    for a in n_indices:
        for b in n_indices - {a}:
            for X in list(combinations(n_indices - {a, b}, m - 1)):
                Y = n_indices - set(X) - {a, b}
                tau = [a] + sorted(list(X)) + [b] + sorted(list(Y))
                inv_tau = {val : i for i, val in enumerate(tau)}

                for psi in product([-1, 1], repeat=m):
                    # We pass 0 as the index 'i' since we are only checking one matrix
                    tasks.append((a, b, psi, G, inv_tau, X, Y, 0))

    print(f"Total LP tasks to solve: {len(tasks)}")

    # 2. Iterate and solve (Single threaded)
    for args in tqdm(tasks):
        val, _ = LP(*args)
        if val is not None:
            if val > max_m_height:
                max_m_height = val
    
    return max_m_height

# --- MAIN ---
if __name__ == "__main__":
    n = 9
    k = 4
    m = 3
    
    G = np.array([[-10.,   6., -29., -24.,   9.],
                    [  3.,   9.,   8., -14., -30.],
                    [ -1., -10.,  23.,  20.,   3.],
                    [ 21.,  12.,   8.,   1., -16.]])
    G = np.hstack((np.eye(k), G))

    # Check dimensions
    if G.shape != (k, n):
        print(f"Error: Matrix G shape {G.shape} does not match k={k}, n={n}")
    else:
        print(f"Calculating m-height for n={n}, k={k}, m={m}...")
        print(f"Matrix G:\n{G}")
        
        result = calculate_m_height(n, k, m, G)
        
        print("\n" + "="*30)
        print(f"Calculated m-height: {result}")
        print("="*30)