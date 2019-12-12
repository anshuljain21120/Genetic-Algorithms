import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 30
values = [11,12,13,14,15,16,17,18,19,10,25,38,55,15,62,
          99,35,12,86,15,69,56,12,98,35,69,86,87,51,29]
weight = [63,52,46,25,45,95,69,53,64,36,78,89,52,69,14,
          87,36,58,79,65,27,81,35,86,93,87,41,65,63,57]
capacity = int(min(weight))*10

cross_over_prob = 0.7
mutation_prob = (1/n)

populationSize = 500
noOfGenerations = 50

noOfOffsprings = populationSize

def validate(chrm,weight,capacity):
    cnt = 0
    for i in range(len(chrm)):
        if cnt <= capacity:
            cnt+=chrm[i]*weight[i]
        if cnt > capacity:
            chrm[i]=0
    return chrm

def fitness(chrm,values):
    cnt = 0
    for i in range(len(chrm)):
        cnt+=chrm[i]*values[i]
    return cnt

def randinit(n):
    chrm = [0]*n
    noOfIteration = random.randint(0,n)
    for i in range(noOfIteration+1):
        chrm[random.randint(0,n)-1] = 1
    return chrm

def flip(chrm,point):
    if chrm[point]==1:
        chrm[point] = 0
    elif chrm[point]==0:
        chrm[point] = 1
    return chrm

def mutation(chrm,n):
    point = random.randint(0,n-1)
    chrm = flip(chrm,point)
    return chrm

def crossover(chrm1,chrm2,n):
    point = random.randint(0,n)
    chrm3 = chrm1[:point]
    chrm3.extend(chrm2[point:])
    chrm4 = chrm2[:point]
    chrm4.extend(chrm1[point:])
    return chrm3,chrm4

def parentSelection(chrm1,chrm2,values):
    if(fitness(chrm1,values)>=fitness(chrm2,values)):
        return chrm1
    return chrm2

def matingPoolCreation(populationSize,population,values):
    matingPool = []
    while len(matingPool)<2:
        rand1 = random.randint(0,populationSize-1)
        rand2 = random.randint(0,populationSize-1)
        parent = parentSelection(population[rand1],population[rand2],values)
        if parent not in matingPool:
            matingPool.append(parent)
    return matingPool

def bestchromosome(population,values):
    p = pd.DataFrame(np.zeros((len(population), 2)), columns=['Chromosome', 'Fitness'], dtype=object)
    for i in range(len(population)):
        p.at[i, 'Chromosome'] = population[i]
        p.at[i, 'Fitness'] = fitness(population[i], values)
    p = p.sort_values(by=['Fitness'], ascending=False)
    return p.iloc[0,0], p.iloc[0, 1]

def plotting(bestFitness):
    plt.plot(bestFitness)
    plt.savefig('Output.png')

print("values = {}\nweight = {}\ncapacity = {}".format(values,weight,capacity))
population = []
while len(population) < populationSize:
    chromosome = randinit(n)
    chromosome = validate(chromosome,weight,capacity)
    if chromosome not in population:
        population.append(chromosome)
print("\n best in population of initial generation is {}".format(bestchromosome(population,values)))

bestFitness = []
for i in range(noOfGenerations):
    offspring = []
    while len(offspring)<noOfOffsprings:

        # Creating mating pool
        matingPool = matingPoolCreation(populationSize,population,values)

        # Applying crossover using crossover probability
        rand = random.random()
        if(rand<=cross_over_prob):
            matingPool[0],matingPool[1] = crossover(matingPool[0],matingPool[1],n)

        # Applying mutation using mutation probability
        rand = random.random()
        if(rand<=mutation_prob):
            matingPool[0] = mutation(matingPool[0],n)
        rand = random.random()
        if(rand<=mutation_prob):
            matingPool[1] = mutation(matingPool[1],n)
        
        for j in matingPool:
            offspring.append(validate(j,weight,capacity))

    # Feeding new generation into population
    population.extend(offspring)
    del(offspring)
    
    # Finding best chromosome of generation
    finalchromosome,bestFit = bestchromosome(population,values)
    f.write("\n-----------------------------------------------------------------" +
            "-------------------------------------------------------------------" +
            str(i+1) + "\tGeneration\n")
    f.write("chromosome {} \t fitness {}\n".format(finalchromosome,bestFit))
    bestFitness.append(bestFit)

print("\n\nbest in population of last generation is {}\nchromosome {} ".format(bestFitness[noOfGenerations-1],finalchromosome))
plotting(bestFitness)