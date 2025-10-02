import numpy as np
import pulp as pl

#this is correct
def rand_perms(n, d):
    perms = []
    for _ in range(n):
        perm = np.random.permutation(d) + 1  # +1 to make it 1-indexed
        perms.append(perm)
    return perms

#this is correct 
def shatter_matrix(perms):
    n = len(perms[0])
    d = len(perms)
    F = np.zeros(((n-1)*d, n))
    for j in range(d):
        for i in range(1, n):
            F[j*(n-1)+(i-1), :] = (perms[j] >= (i+1)).astype(float)
    return F


#this gives correct answer on [[1,2,3],[3,2,1]] and [[1,2,3],[1,2,3]]
def check_shatter(F, n, d, eps):
    # Primary LP
    prob = pl.LpProblem("ShatterCheck", pl.LpMaximize)
    w = [pl.LpVariable(f"w_{k}", lowBound=eps) for k in range((n-1)*d)]
    b = pl.LpVariable("b", lowBound=0)
    # Objective is zero (feasibility problem)
    prob += 0

    # F' * w == b * ones(n)
    for i in range(n):
        prob += pl.lpSum(F[k, i] * w[k] for k in range((n-1)*d)) == b

    print(prob)
    # Solve
    prob.solve()
    if pl.LpStatus[prob.status] == "Optimal":
        return (True, "weights", [pl.value(var) for var in w])
    else:
        # Dual LPs
        for i in range(n):
            dual = pl.LpProblem(f"Dual_{i}", pl.LpMinimize)
            z = [pl.LpVariable(f"z_{j}") for j in range(n)]
            # z' * ones(n) <= 0
            dual += pl.lpSum(z) <= 0
            # F * z >= 0
            for k in range((n-1)*d):
                dual += pl.lpSum(F[k, j] * z[j] for j in range(n)) >= 0
            # z[i] >= 1.0
            dual += z[i] >= 1.0
            # Objective is zero (feasibility)
            dual += 0
            dual.solve()
            if pl.LpStatus[dual.status] == "Optimal":
                return (False, "obstruction", [pl.value(var) for var in z])
        return (False, "indeterminate", [0])
