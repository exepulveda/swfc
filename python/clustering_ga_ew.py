import sys
import logging
import collections
import math
import sys

import clusteringlib as cl
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from copy import deepcopy

from cluster_utils import fix_weights

class CentroidIndividual(object):
    def __init__(self,NC,ND,min_values,max_values):
        self.NC = NC
        self.ND = ND
        self.min_values = min_values
        self.max_values = max_values
        self.centroid = np.empty((NC,ND))
        self.fitness = creator.FitnessMin()
        self.u = None

    def clone(self):
        new_obj = CentroidIndividual(self.NC,self.ND,self.min_values,self.max_values)
        new_obj.fitness = deepcopy(self.fitness)  
        new_obj.centroid[:,:] = self.centroid      
        new_obj.u = self.u.copy()
        
        return new_obj
        
    def check(self):
        for i in range(self.NC):
            for j in range(self.ND):
                if not(self.min_values[j] <= self.centroid[i,j] <= self.max_values[j]):
                    print(i,j,self.centroid[i,j])
                    assert False, "PROBLEM 2222"
        
    def __deepcopy__(self,memo):
        return self.clone()        
        
def clone_centroid(sol):
    return sol.clone()

def create_ga_centroids(data,weights,m,lambda_value,types,cat_values):
    """
    Helper method to configure evolution algorithm.

    Uses toolbox from deap and potentially could use scoop to distribute or
    parallelize.
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    toolbox = base.Toolbox()

    toolbox.register("clone", clone_centroid)
    toolbox.register("mate", crossover_centroid)
    toolbox.register("evaluate", evaluate_centroid,data=data,weights=weights,m=m,lambda_value=lambda_value)
    toolbox.register("mutate", mutate_centroid,types=types,cat_values=cat_values)
    toolbox.register("select", tools.selTournament, tournsize=9)

    return toolbox
    
def mutate_centroid(ind,types,cat_values):
    new_solution = ind.centroid
    
    NC,ND = new_solution.shape

    #select at random a cluster to modify
    sel_cluster = np.random.randint(NC)
    #select at random a dimension to modify
    sel_dim = np.random.randint(ND)

    new_solution[sel_cluster,sel_dim] += np.random.normal(loc=0.0, scale=0.1)
    new_solution[sel_cluster,sel_dim] =  np.clip(new_solution[sel_cluster,sel_dim],ind.min_values[sel_dim],ind.max_values[sel_dim])

    return ind,
    
def crossover_centroid(ind1,ind2):
    ind1.check()
    ind2.check()
    
    child1, child2 = tools.cxUniform(ind1.centroid.flatten(),ind2.centroid.flatten(),indpb=0.5)
    
    ind1.centroid[:,:] = child1.reshape((ind1.NC,ind1.ND))
    ind2.centroid[:,:] = child2.reshape((ind2.NC,ind2.ND))
    
    ind1.check()
    ind2.check()
    
    return ind1, ind2

def similar_op_centroid(ind1,ind2):
    return np.array_equal(ind1.centroid,ind2.centroid)

def evolve_centroids(toolbox,initial_centroids,values,min_values,max_values,npop,ngen,stop_after,cxpb,mutpb,verbose=False):
    NC,ND = initial_centroids.shape
    N = len(values)
    
    pop = [CentroidIndividual(NC,ND,min_values,max_values) for _ in range(npop)]
    for i,ind in enumerate(pop):
        if i == 0:
            ind.centroid[:,:] = initial_centroids
        else:
            #random
            for k in range(NC):
                j = np.random.choice(N)
                ind.centroid[k,:] = values[j,:]
                
        ind.check()
            
    hof = tools.HallOfFame(1,similar=similar_op_centroid)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "avg", "max", "best"

    #logger.info("Evaluating initial population")
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    
    record = stats.compile(pop) if stats else {}

    logbook.record(gen=0,best=hof[0].fitness.values[0], **record)
    if verbose: print(logbook.stream)

    evaluations = 0
    no_improvements = 0

    #logger.info("Starting evolution!")
    for gen in range(1, ngen+1):
        prev_fitness = hof[0].fitness.values[0]
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        evals_gen = len(invalid_ind)

        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)

        current_fitness = hof[0].fitness.values[0]

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen,best=hof[0].fitness.values[0], **record)
        #logger.info(logbook.stream)
        if verbose: print(logbook.stream)
        evaluations += evals_gen

        if current_fitness >= prev_fitness: #worse fitness
            no_improvements += 1
        else:
            no_improvements = 0
            
        if no_improvements > stop_after:
            break

    return pop, stats, hof, logbook, gen, evaluations
    
    
def evaluate_centroid(ind,data,weights,m,lambda_value,verbose=0):
    centroid = ind.centroid
    
    ret = evaluate(centroid,data,weights,m,lambda_value,verbose=0)
    ind.u = ret[1]
    return ret
    
def evaluate(centroid,data,weights,m,lambda_value,C=5.0,verbose=0):
    N,ND = data.shape
    ND2 = len(weights)
    NC,ND3 = centroid.shape
    
    assert ND2 == ND,"weigths and data shape: %d, %d"%(ND2,ND)
    assert ND2 == ND3,"weigths adn centroid shape: %d, %d"%(ND2,ND3)

    
    u = np.asfortranarray(np.empty((N,NC)),dtype=np.float32)
    centroid_F = np.asfortranarray(centroid,dtype=np.float32)
    data_F = np.asfortranarray(data,dtype=np.float32)
    weights_F = np.asfortranarray(weights,dtype=np.float32)

    cl.clustering.update_membership_ew(data_F,centroid_F,m,u,weights_F,verbose)
    #centroid,data,m,u,weights,verbose
    verbose = 0
    
    if verbose > 2:
        print("centroid",centroid)
        print("m",m)
        print("weights",weights)
        print("lambda_value",lambda_value)

    index1,jm,min_cluster_distance = cl.clustering.compactness_weighted_ew(data_F,centroid_F,m,u,weights_F,1,lambda_value,verbose)
    
    #print('compactness',index1,'separation',min_cluster_distance)
    
    #return index1 + 100.0/min_cluster_distance,jm,u
    #return index1/min_cluster_distance,jm,u
    if min_cluster_distance != 0.0:
        return jm + C/min_cluster_distance,u,index1,jm,min_cluster_distance
    else:
        return jm + C,u,index1,jm,min_cluster_distance
    
def optimize_centroids(data,current_solution,weights,m,lambda_value,types,cats,ngen=50,npop=50,cxpb=0.8,mutpb=0.2,stop_after=5,min_values=None,max_values=None,reg=2,verbose=False):
    '''
    Optimize by PSO the centroids given weights
    '''
    logger = logging.getLogger("optimize_centroids")
    if min_values is None:
        min_values = np.min(data,axis=0)
    
    if max_values is None:
        max_values = np.max(data,axis=0)
    
    #if verbose > 0: print('min_values',min_values)
    #if verbose > 0: print('max_values',max_values)

    c1 = 1.49
    c2 = 1.49
    w = 0.72
    
    ND = len(weights.shape)
    
    N = len(data)
    
    toolbox = create_ga_centroids(data,weights,m,lambda_value,types,cats)
    pop, stats, hof, logbook, gen, evaluations = evolve_centroids(toolbox,current_solution,data,min_values,max_values,npop,ngen,stop_after,cxpb,mutpb,verbose=verbose)

    best_ga = hof[0]

    return best_ga.centroid,best_ga.u,best_ga.fitness.values[0],best_ga.fitness.values[0],0,evaluations


#======================================
class WeightIndividual(object):
    def __init__(self,ND,force):
        self.ND = ND
        self.force = force
        self.weights = np.empty(ND)
        self.fitness = creator.FitnessMin()
        self.u = None

    def clone(self):
        new_obj = WeightIndividual(self.ND,self.force)
        new_obj.fitness = deepcopy(self.fitness)  
        new_obj.weights[:] = self.weights      
        new_obj.u = self.u.copy()
        
        return new_obj
        
    def check(self):
        for j in range(self.ND):
            if not(0.0 <= self.weights[j] <= 1.0):
                print(j,self.weights[j])
                assert False, "PROBLEM 2222"
        
    def __deepcopy__(self,memo):
        return self.clone()        
        
def clone_weights(sol):
    return sol.clone()
    
def evaluate_weights(ind,data,centroid,m,lambda_value,verbose=0):
    weights = ind.weights
    
    ret =  evaluate(centroid,data,weights,m,lambda_value,verbose=0)
    ind.u = ret[1]
    
    return ret


def create_ga_weights(data,centroids,initial_weights,m,lambda_value,force):
    """
    Helper method to configure evolution algorithm.

    Uses toolbox from deap and potentially could use scoop to distribute or
    parallelize.
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    toolbox = base.Toolbox()

    toolbox.register("clone", clone_weights)
    toolbox.register("mate", crossover_weights,force=force)
    toolbox.register("evaluate", evaluate_weights,data=data,centroid=centroids,m=m,lambda_value=lambda_value)
    toolbox.register("mutate", mutate_weights,force=force)
    toolbox.register("select", tools.selTournament, tournsize=9)

    return toolbox
    
def mutate_weights(ind,force):
    new_solution = ind.weights.copy()
    
    ND = len(new_solution)

    #select at random a dimension to modify
    sel_dim = np.random.randint(ND)

    new_solution[sel_dim] += np.random.normal(loc=0.0, scale=0.1)
    new_solution[sel_dim] = np.clip(new_solution[sel_dim],0.0000001,1.0)
    new_solution[:] =  fix_weights(new_solution[:],force)

    ind.weights[:] = new_solution
    
    return ind,
    
def crossover_weights(ind1,ind2,force):
    #ind1.check()
    #ind2.check()
    
    child1, child2 = tools.cxUniform(ind1.weights,ind2.weights,indpb=0.5)
    
    ind1.weights[:] = fix_weights(child1,force)        
    ind2.weights[:] = fix_weights(child2,force)        
    
    #ind1.check()
    #ind2.check()
    
    return ind1, ind2

def similar_op_weights(ind1,ind2):
    return np.array_equal(ind1.weights,ind2.weights)

def evolve_weights(toolbox,centroids,initial_weights,values,ngen,npop,stop_after,cxpb,mutpb,force,verbose=False):
    NC,ND = centroids.shape
    N = len(values)
    
    pop = [WeightIndividual(ND,None) for _ in range(npop)]
    for i,ind in enumerate(pop):
        if i == 0:
            ind.weights[:] = initial_weights
        else:
            #random
            ind.weights[:] = np.random.random(ND)
            ind.weights[:] /= np.sum(ind.weights[:])
            ind.weights[:] = fix_weights(ind.weights[:],force=force)
                
        ind.check()
            
    hof = tools.HallOfFame(1,similar=similar_op_weights)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "min", "avg", "max", "best"

    #logger.info("Evaluating initial population")
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    
    record = stats.compile(pop) if stats else {}

    logbook.record(gen=0,best=hof[0].fitness.values[0], **record)
    if verbose: print(logbook.stream)

    evaluations = 0
    no_improvements = 0
    
    #logger.info("Starting evolution!")
    for gen in range(1, ngen+1):
        prev_fitness = hof[0].fitness.values[0]
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        evals_gen = len(invalid_ind)

        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen,best=hof[0].fitness.values[0], **record)
        #logger.info(logbook.stream)
        if verbose: print(logbook.stream)

        evaluations += evals_gen

        current_fitness = hof[0].fitness.values[0]

        if current_fitness >= prev_fitness:
            no_improvements += 1
        else:
            no_improvements = 0
            
        if no_improvements > stop_after:
            break
            
    return pop, stats, hof, logbook, gen, evaluations
    
def optimize_weights(data,centroids,current_solution,m,lambda_value,ngen=50,npop=50,cxpb=0.8,mutpb=0.2,stop_after=5,force=None,verbose=False):
    '''
    Optimize by PSO the centroids given weights
    '''
    NC,ND = centroids.shape
    
    N = len(data)
    
    toolbox = create_ga_weights(data,centroids,current_solution,m,lambda_value,force)
    pop, stats, hof, logbook, gen, evaluations = evolve_weights(toolbox,centroids,current_solution,data,ngen,npop,stop_after,cxpb,mutpb,force,verbose=verbose)

    best_ga = hof[0]

    return best_ga.weights,best_ga.u,best_ga.fitness.values[0],evaluations
