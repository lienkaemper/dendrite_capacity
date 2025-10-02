from src.dendrites import shatter_matrix, rand_perms, check_shatter
import matplotlib.pyplot as plt
import numpy as np
import pulp as pl

solver_list = pl.listSolvers()
print("Available solvers:", solver_list)

perms = rand_perms(5, 3)
print("Permutations:")
print(np.array(perms))
F = shatter_matrix(perms)
print("Shatter matrix:")
print(F)

plt.imshow(F)
plt.title("Shatter matrix")
#plt.show()

x = pl.LpVariable("x", 0, 3)
y = pl.LpVariable("y", cat="Binary")
prob = pl.LpProblem("myProblem", pl.LpMinimize)
prob += x + y <= 2
prob += -4*x + y
status = prob.solve()

print("Status:", pl.LpStatus[status])
print("x =", pl.value(x))
print("y =", pl.value(y))
print("Objective =", pl.value(prob.objective))

perms = [np.array([1,2,3]), np.array([3,2,1])]
F = shatter_matrix(perms)
print("Shatter matrix for perms [[1,2,3],[3,2,1]]:")
print(F)
result = check_shatter(F, n=3, d=2, eps=0.1)
print("Check shatter result:", result)

#expected output: (True, 'weights', [0.1, 0.1, 0.1, 0.1])

perms = [np.array([1,2,3]), np.array([1,2,3])]
F = shatter_matrix(perms)
print("Shatter matrix for perms [[1,2,3],[1,2,3]]:")
print(F)
result = check_shatter(F, n=3, d=2, eps=0.1)
print("Check shatter result:", result)

#expected output: (False, 'obstruction', [1.0, -1.0, 0.0])