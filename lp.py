from itertools import combinations
import multiprocessing

import pulp

def LP(S, j, G, k, n):
    """
    Solves the LP for a specific subset S and target index j.
    """
    # S_bar contains all indices in [0, n-1] that are NOT in S
    S_set = set(S)
    S_bar = [t for t in range(n) if t not in S_set]

    # Initialize the LP problem
    lp_problem = pulp.LpProblem("LP_S_j", pulp.LpMaximize)

    # Define k real-valued variables (unbounded)
    u = [pulp.LpVariable(f"u_{i}", cat=pulp.LpContinuous, lowBound=None, upBound=None) for i in range(k)]

    # Objective: maximize sum(g_{i,j} * u_i)
    lp_problem += pulp.lpSum([G[i, j] * u[i] for i in range(k)])

    # Constraints: -1 <= sum(g_{i,t} * u_i) <= 1 for all t in S_bar
    for t in S_bar:
        expr = pulp.lpSum([G[i, t] * u[i] for i in range(k)])
        lp_problem += expr <= 1
        lp_problem += expr >= -1

    # Solve quietly
    lp_problem.solve(pulp.HiGHS_CMD(msg=False))

    # Handle statuses
    if lp_problem.status == pulp.LpStatusInfeasible:
        return 0
    elif lp_problem.status == pulp.LpStatusUnbounded:
        return float('inf')
    elif lp_problem.status == pulp.LpStatusOptimal:
        return lp_problem.objective.value()
    else:
        return 0

def LP_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing."""
    return LP(*args)

def calculate_m_height_parallel(n, k, m, G):
    """
    Generates all tasks and runs LP in parallel to find the max m-height.
    """
    tasks = []
    
    # Generate all subsets S of size m from [0, n-1]
    for S in combinations(range(n), m):
        for j in S:
            tasks.append((S, j, G, k, n))

    if not tasks:
        return 1.0

    total_tasks = len(tasks)
    max_m_height = float('-inf')
    
    chunk_size = max(1, total_tasks // (multiprocessing.cpu_count() * 4))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.imap_unordered(LP_wrapper, tasks, chunksize=chunk_size)
        
        for val in results:
            if val is not None and val > max_m_height:
                max_m_height = val

    return max_m_height

def calculate_m_height(n, k, m, G):
    """
    Generates all tasks and runs LP in a single process to find the max m-height.
    """
    max_m_height = float('-inf')
    tasks_exist = False
    
    # Generate all subsets S of size m from [0, n-1]
    for S in combinations(range(n), m):
        for j in S:
            tasks_exist = True
            
            # Call the LP function directly
            val = LP(S, j, G, k, n)
            
            # Update the maximum m-height found so far
            if val is not None and val > max_m_height:
                max_m_height = val

    # Return 1.0 if no tasks were generated, matching the parallel version's fallback
    if not tasks_exist:
        return 1.0

    return max_m_height