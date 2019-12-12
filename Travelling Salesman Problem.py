import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def fitness(chrm,distances):
    dis = 0
    for i in range(len(chrm)-1):
        dis += distances[chrm[i]][chrm[i+1]]
    
    dis += distances[chrm[0]][chrm[len(chrm)-1]]
    
    return dis
def randinit(n,lenchrm,dis):
    df = pd.DataFrame(np.zeros((n,2)),columns=["Chromosome","Fitness"], dtype=object)
    i = 0
    arr = list(range(lenchrm))
    while i < n:
        random.shuffle(arr)
        if arr not in list(df["Chromosome"]):
            df.at[i,"Chromosome"] = arr.copy()
            df.at[i,"Fitness"] = fitness(arr,dis)
            i = i+1
    df["Fitness"] = df["Fitness"].astype('int64')
    return df
def distances(No_Of_Cities):
    
    distance = np.zeros((No_Of_Cities,No_Of_Cities))
    
    for city in range(No_Of_Cities):
        cities = [i for i in range(No_Of_Cities) if not i==city]
        for to_city in cities:
            if(distance[city][to_city]==0 and distance[to_city][city]==0):
                distance[city][to_city] = distance[to_city][city] = random.randint(1000,100000)

    return distance
def crossover(chrm1,chrm2):
    n = len(chrm1)
    chrm3 = chrm1[:int(n/2)]
    for i in range(int(n/2), int(n/2)+n):
        if chrm2[i%n] not in chrm3:
            chrm3.append(chrm2[i%n])
    n = len(chrm2)
    chrm4 = chrm2[:int(n/2)]    
    for i in range(int(n/2), int(n/2)+n):
        if chrm1[i%n] not in chrm4:
            chrm4.append(chrm1[i%n])
    
    return chrm3, chrm4
def mutation(chrm):
    if random.random() < 1/len(chrm):
        n = len(chrm)
        a = random.randint(0, n-1)
        b = random.randint(0, n-1)
        while b==a:
            b = random.randint(0, n-1)
        chrm[a], chrm[b] = chrm[b], chrm[a]
    return chrm
def giveParent(pop):
    n = pop.shape[0]
    parents = pd.DataFrame(np.zeros((5,2)),columns=['Parents',"Fitness"], dtype=object)
    i = 0
    while len(parents) <= 5:
        r = random.randint(0, n-1)
        parent = pop.iloc[r, 0]
        fitness = pop.iloc[r,1]
        if parent not in list(parents["Parents"]):
            parents.at[i,"Parents"] = parent
            parents.at[i,"Fitness"] = fitness
            i = i+1
    
    parents = parents.sort_values(by=['Fitness'])
    return parents.iloc[0, 0], parents.iloc[1, 0]
def replaceWorst(p, dis, v=False):
    pop = p.copy()
    p1, p2 = giveParent(pop)
    c1, c2 = crossover(p1, p2)
    c1 = mutation(c1)
    c2 = mutation(c2)
    if v:
        print('Child 1:{0}\tFitness: {1}\nChild 2:{2}\tFitness: {3}'.format(c1, fitness(c1,dis), c2, fitness(c2,dis)))
    
    pop = pop.sort_values(by=['Fitness'])
    pop = pop.reset_index(drop=True)
    
    if c1 not in list(pop["Chromosome"]):
        pop.loc[p.shape[0]-2] = [c1, fitness(c1,dis)]
    
    if c2 not in list(pop["Chromosome"]):
        pop.loc[p.shape[0]-1] = [c2, fitness(c2,dis)]
    
    return pop
def bestFitness(pop):
    return min(list(pop['Fitness']))
def bestChromosome(pop):
    p = pop.sort_values(by=['Fitness'])
    return p.iloc[0, 0]

N = int(input('Value of N: '))
pop_size = int(input('Population Size: '))
show_progress = False
NUMBER_OF_CITIES = N;

dis = distances(N)

generationInfo = pd.DataFrame(np.zeros((10000, 2)), columns=['Chromosome', 'Fitness'], dtype=object)

for generation in range(10000):
    if generation == 0:
        pop = randinit(pop_size, N, dis)
    
    generationInfo.loc[generation] = [bestChromosome(pop), bestFitness(pop)]
    
    if show_progress:
        print("\nGeneration {}".format(generation))
    pop = replaceWorst(pop,dis,show_progress)
    
Optimal_Solution = [bestChromosome(pop), bestFitness(pop)]
print(Optimal_Solution)

import matplotlib.pyplot as plt
plt.plot(range(generationInfo.shape[0]), list(generationInfo['Fitness']))