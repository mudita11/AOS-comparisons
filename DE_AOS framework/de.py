#!/usr/bin/env python

try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
import random
import math
import csv
from numpy.linalg import inv

import aos

# chunk is popsize
# F1 child fitness
# X parent population
# u offspring pop
# index: best candidate in current pop
# f_min = fitness minimum
# x_min = position minimum

def DE(fun, lbounds, ubounds, budget, FF, CR, alpha, p_min, W, C, problem_index):
    
    def rand1(population, samples, scale): # DE/rand/1
        r0, r1, r2 = samples[:3]
        return (population[r0] + scale * (population[r1] - population[r2]))

    def rand2(population, samples, scale): # DE/rand/2
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

    def rand_to_best2(population, samples, scale): # DE/rand-to-best/2
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

    def current_to_rand1(population, samples, scale): # DE/current-to-rand/1
        r0, r1, r2 = samples[:3]
        return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))

    def select_samples(popsize, candidate, number_samples):
        """
        obtain random integers from range(popsize),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(popsize))
        idxs.remove(candidate)
        return(np.random.choice(idxs, 5, replace = False))



    iteration = 0
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    NP = 10 * dim
    chunk = NP
    X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
    F = [fun(x) for x in X];
    budget -= chunk
    
    u = [[0 for j in range(int(dim))] for i in range(int(chunk))];#print(u)
    F1 = np.zeros(int(chunk));
    
    index = np.argmin(F);
    if f_min is None or F[index] < f_min:
        x_min, f_min = X[index], F[index];
    best_so_far = f_min
    best_so_far1 = best_so_far

    n_operators = 4
    aos_method = aos.Rec_PM(chunk, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops = n_operators, alpha = alpha, p_min = p_min, scaling_factor = C, window_size = W)

    mutations = [rand1, rand2, rand_to_best2, current_to_rand1]

    while budget > 0:
        
        fill_points = np.random.randint(dim, size = NP)
        
        for i in range(NP):
            SI = aos_method.Selection()
            assert SI >= 0 and SI <= len(mutations)
            mutate = mutations[SI]
            aos_method.opu[i] = SI
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            best = np.argmin(aos_method.F)
            crossovers = (np.random.rand(dim) < CR)
            crossovers[fill_points[i]] = True
            trial = aos_method.X[i]
            bprime = mutate(aos_method.X, r, FF)
            aos_method.u[i][:] = np.where(crossovers, bprime, trial)
    
        aos_method.F1 = [fun(x) for x in aos_method.u]

        aos_method.AOSUpdate()

        index = np.argmin(aos_method.F)
        if aos_method.f_min is None or aos_method.F[index] < aos_method.f_min:
            aos_method.x_min, aos_method.f_min = aos_method.X[index], aos_method.F[index]
        aos_method.best_so_far1 = aos_method.f_min;
        if aos_method.best_so_far1 < aos_method.best_so_far:
            aos_method.best_so_far = aos_method.best_so_far1
        iteration = iteration+1
        budget -= chunk
    
    return aos_method.best_so_far

