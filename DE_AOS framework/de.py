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


# ===============================================
# prepare (the most basic example solver)
# ===============================================
'''
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""    
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(min([budget, max_chunk_size]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim) #generate chunkXdim size matrix
        F = [fun(x) for x in X] #F is a vector of size X where for each x in X we get F[X] 
        if fun.number_of_objectives == 1:
            index = np.argmin(F)
            if f_min is None or F[index] < f_min:
                x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min
'''

def DE(fun, lbounds, ubounds, budget, FF, CR, alpha, p_min, W, C, problem_index):
    
    def rand1(population, samples, scale): # DE/rand/1
        r0, r1, r2 = samples[:3]
        return (population[r0] + scale * (population[r1] - population[r2]))

    def rand2(population, samples, scale): # DE/rand/2
        r0, r1, r2, r3, r4 = samples[:5]
        return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

    # FIXME: Is this correct? The implementation in scipy does something different!
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
    # chunk is popsize
    # F1 child fitness
    # X parent population
    # u offspring pop
    # index: best candidate in current pop
    # f_min = fitness minimum
    # x_min = best_so_far
    # n_improvements = 0
    n_operators = 4
    aos_method = aos.PM_AdapSS(chunk, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops = n_operators, alpha = alpha, p_min = p_min, scaling_factor = C, window_size = W)

    mutations = [rand1, rand2, rand_to_best2, current_to_rand1]

    #f=open("statistics","a+")
    #f.write(str(problem_index))
    #f.write("\n")
    
    while budget > 0:
        
        fill_points = np.random.randint(dim, size = NP)
        
        # FIXME: This loop can be implemented as matrix operation.
        for i in range(NP):
            SI = aos_method.Selection(); #f.write(str(SI))
            assert SI >= 0 and SI <= len(mutations)
            mutate = mutations[SI]
            aos_method.opu[i] = SI
            # No mutation strategy needs more than 5.
            r = select_samples(NP, i, 5)
            best = np.argmin(aos_method.F);#print("best",best)
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

        #if iteration%100==0:
            #print("iteration",iteration)
            #print("r: ",aos_method.reward)
            #print("Q: ",aos_method.Q)
            #print("probability: ",aos_method.p)
            #print("operator: ",aos_method.operator)
            #print("window_delta_f",aos_method.window_delta_f)
            #print("tran matrix: ",aos_method.tran_matrix)

        #f.write("\n")
        #f.write(str(budget))
        #f.write("\n")
        #f.close()
        
        iteration = iteration+1
        budget -= chunk
        #print("iteration :",iteration)
    return aos_method.best_so_far


#def rec_PM(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
    #lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    #dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    #iteration = 0;
    #NP=10*dim
    #p=np.zeros(int(strategy)); # empirical quality estimate
    #for i in range(strategy):
        #p[i]=1.0/strategy
    
    #last_p=np.zeros(int(strategy))
    #Q=np.zeros(int(strategy)); # empirical quality estimate
    #for i in range(strategy):
        #Q[i]=1.0
    
    #reward=np.zeros(strategy);rewardt1=np.zeros(strategy);
    
    #chunk = NP
    #X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
    #F=[fun(x) for x in X]; #print("F",F) # Evaluate the fitness for each individual
    #budget-=chunk
    
    #u=[[0 for j in range(int(dim))] for i in range(int(chunk))];#print(u)
    #F1=np.zeros(int(chunk));
        
    #index = np.argmin(F);
    #if f_min is None or F[index] < f_min:
        #x_min, f_min = X[index], F[index];
    #print("x_min",x_min)
    #best_so_far=f_min;

    #while budget > 0:
    
        #if a_list:    # checks if a is not empty;if this is true then enter
            #print("a1",a_list)
            #SI=random.choice(a_list) #a[random.randint(0,len(a)-1] also works
            #print("SI",SI)    #a=list(a)
            #a_list.remove(SI)   #print("a after removing ",a_list)
        #else:
            #print(p)
            #SI=np.argmax(np.random.multinomial(1,p,size=1))
        
        #for i in range(chunk):
            #r1=random.randint(0,chunk-1); #print("r1",r1)# Select uniform randomly r1!=r2!=r3!=i
            #r2=random.randint(0,chunk-1); #print("r2",r2)
            #r3=random.randint(0,chunk-1);#print("r3",r3)
            #r4=random.randint(0,chunk-1);#print("r4",r4)
            #r5=random.randint(0,chunk-1); #print("r5",r5)
            #best=np.argmin(F);#print("best",best)
            #while r1==i:
                #r1=random.randint(0,chunk-1)
            #while r2==i or r2==r1:
                #r2=random.randint(0,chunk-1)
            #while r3==i or r3==r1 or r3==r2:
                #r3=random.randint(0,chunk-1)
            #while r4==i or r4==r1 or r4==r2 or r4==r3:
                #r4=random.randint(0,chunk-1)
            #while r5==i or r5==r1 or r5==r2 or r5==r3 or r5==r4:
                #r5=random.randint(0,chunk-1)
            #jrand = random.randint(0,dim-1);
            #for j in range(dim):
                #print("i,j",i,j)
                #if random.random()<CR or j==jrand:
                    #if SI==0:
                        #u[i][j]=X[r1][j]+FF*(X[r2][j]-X[r3][j]) # DE/rand/1
                    #elif SI==1:
                        #print(i,r1,r2,r3,r4,r5)
                        #u[i][j]=X[r1][j]+FF*(X[r2][j]-X[r3][j])+FF*(X[r4][j]-X[r5][j]) # DE/rand/2
                    #elif SI==2:
                        #print(i,r1,r2,r3,r4,r5)
                        #u[i][j]=X[r1][j]+FF*(X[best][j]-X[r1][j])+FF*(X[r2][j]-X[r3][j])+FF*(X[r4][j]-X[r5][j]) # DE/rand-to-best/2
                    #elif SI==3:
                        #print(i,r1,r2,r3)
                        #u[i][j]=X[i][j]+FF*(X[r1][j]-X[i][j])+FF*(X[r2][j]-X[r3][j]) # DE/current-to-randr/1
                #else:
                    #print("i,j1",i,j)
                    #u[i][j]=X[i][j]
        #F1=[fun(x) for x in u]; #print("F1",F1)# Evaluate the offspring ui
        #print(F,F1);
       
        #counter=0
        #for i in range(chunk):
            #if F1[i]<=F[i]:
                #counter=counter+1
                #for j in range(dim):
                    #X[i][j]=u[i][j]
                #F[i]=F1[i]
        #print("Counter: ",counter)
        #index=np.argmin(F)
        #if f_min is None or F[index] < f_min:
            #x_min, f_min = X[index], F[index]
        
        #best_so_far1=f_min;
        #if best_so_far1<best_so_far:
            #best_so_far=best_so_far1
    
        #print("iteration,last SI,SI: ",iteration,last_SI,SI)
       
        #Application of one strategy might affect the others
        #CreditAssignment.GetReward(i)
        #rewardt1=reward;
        #for i in range(strategy):
            #if i==SI:
                #reward[SI]=(counter/chunk)+0.5*rewardt1[i]
            #else:
                #reward[i]=0.5*rewardt1[i]
        #print("reward: ",reward)
        
        
        #Q=np.matmul(np.array(reward),np.linalg.pinv(np.array((1-alpha*np.array(tran_matrix)))));
        #print("Q: ",Q)
        #Q=Q-np.max(Q)
        #print("Q1: ",Q)
        #Q=np.exp(Q)
        #print("Q2: ",Q)
        #sum_Q=np.sum(Q);Q=Q/sum_Q
        #print("Q3: ",Q)
        
        #last_p=p;
        #for i in range(strategy):
            #if np.sum(Q)!=0:
                #p[i]=p_min+(1-strategy*p_min)*(Q[i]/np.sum(Q))

        #for i in range(strategy):
            #for j in range(strategy):
                #tran_matrix[i][j]=last_p[i]+p[j]
        #if iteration>=1:
            #tran_matrix[last_SI][SI]=tran_matrix[last_SI][SI]+1
            
        #tran_matrix[:]=tran_matrix/np.sum(tran_matrix,axis=1)[:,None]
        #if iteration%100==0:
            #print("transi",tran_matrix)
            
        #tran_matrix[:]=tran_matrix/np.sum(tran_matrix)
        
        #tm=np.sum(tran_matrix,axis=1)
        #for i in range(strategy):
        #    for j in range(strategy):
        #        tran_matrix[i][j]=tran_matrix[i][j]/tm[i]

        #print("SI,p: ",SI,p)
        #if (iteration % 100)==0:
            #print("p: ",p)
            #print("tm: ",tm)
            #print("tran_matrix: ",tran_matrix)
        #print("-------------------------------------------------------------------------------")
            
        #iteration=iteration+1;
        #budget-=chunk;
    #print("x_min",x_min)
    #cost = best_so_far;
    #print(cost)
    #return x_min
