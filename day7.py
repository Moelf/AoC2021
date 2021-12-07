import numpy as np
from typing import Any, Callable
from numbers import Number

def cost_p1(crab_loc : np.ndarray, loc : float) -> float:
    return np.sum(np.abs(crab_loc - loc))

def cost_p2(crab_loc : np.ndarray, loc : Any) -> float:
    if isinstance(loc, Number):
        loc = np.array([loc])
    diff = np.abs(crab_loc[:, np.newaxis] - loc[np.newaxis, :])
    return np.sum((1 + diff) * diff / 2, axis=0)

def grad_p1(crab_loc : np.ndarray, soln : float) -> float:
    return np.mean(np.sign(crab_loc - soln))

def min_hori_dist(crab_loc : np.ndarray, grad_fn : Callable) -> float:
    '''
    Momentum to account for L1 non-differentiability :)
    '''

    cont_soln = np.mean(crab_loc) # L2 solution 
    grad = 0
    last_grad = 0
    gamma = 0.9

    max_iter = 1000
    for i in range(max_iter):
        curr_grad = grad_fn(crab_loc, cont_soln)
        grad = grad * gamma + curr_grad
        cont_soln = cont_soln + grad

        if i > 50:
            # Early Termination
            if np.sign(last_grad) != np.sign(grad):
                break
        last_grad = grad

    return np.round(cont_soln)

def part2(crab_loc : np.ndarray) -> float:
    min_d = np.mean(crab_loc) # Exact solution, since it's a L2 cost now
    d = np.array([np.ceil(min_d), np.floor(min_d)])
    return np.min(cost_p2(crab_loc, d))

fname = "input7"
crab_loc = np.loadtxt(fname, delimiter=",")
min_loc = min_hori_dist(crab_loc, grad_p1)
print(f"Part1 index: {min_loc}, cost: {cost_p1(crab_loc, 362)}")
print(f"Part2 cost: {part2(crab_loc)}")
