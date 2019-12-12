import pandas as pd
import numpy as np
import random
graph = [[0,1,1,1,1,0,0,0,0,0,0,0],
         [1,0,1,1,0,0,1,0,0,0,0,0],
         [1,1,0,0,1,1,0,0,0,0,0,0],
         [1,1,0,0,1,0,1,0,0,0,0,0],
         [1,0,1,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,1,1,0,0,0,0],
         [0,1,0,1,0,1,0,1,0,0,0,0],
         [0,0,0,0,0,1,1,0,1,0,0,0],
         [0,0,0,0,0,0,0,1,0,1,1,0],
         [0,0,0,0,0,0,0,0,1,0,1,1],
         [0,0,0,0,0,0,0,0,1,1,0,1],
         [0,0,0,0,0,0,0,0,0,1,1,0]]
colour = ["C1","C2","C3"]
def fitness(chrm,graph):
    coll = 0
    for i in range(len(chrm)):
        for j in range(len(chrm)):
            if graph[i][j] == 1:
                if chrm[i]==chrm[j]:
                    coll = coll+1
    return coll
def randinit(n,graph,colour):
    df = pd.DataFrame(np.zeros((n,2)),columns=["Chromosome","Fitness"], dtype=object)
    i = 0
    chrmlen = len(graph)
    cl = len(colour)
    while i < n:
        chrm = []
        for j in range(chrmlen):
            chrm.append(colour[random.randint(0,cl-1)])
        if chrm not in list(df["Chromosome"]):
            df.at[i,"Chromosome"] = chrm.copy()
            df.at[i,"Fitness"] = fitness(chrm,graph)
            i = i+1
    df["Fitness"] = df["Fitness"].astype('int64')
    return df
def crossover(chrm1,chrm2):
    n = len(chrm1)
    point = random.randint(0,n)
    chrm3 = chrm1[:point]
    chrm3.extend(chrm2[point:])
    chrm4 = chrm2[:point]
    chrm4.extend(chrm1[point:])
    return chrm3,chrm4
def mutation(chrm):
    n = len(chrm)
    point1 = random.randint(0,n-1)
    point2 = random.randint(0,n-1)
    while(point1==point2):
        point1 = random.randint(0,n-1)
    chrm[point2],chrm[point1] = chrm[point1],chrm[point2]
    return chrm.copy()
def parentsel(chrmlist):
    p1 = chrmlist[random.randint(0,99)]
    p2 = chrmlist[random.randint(100,499)]
    return p1,p2
def replaceWorst(pop,graph,v=False):
    pop = pop.sort_values(by=['Fitness'])
    p1, p2 = parentsel(list(pop["Chromosome"]))
    c1, c2 = crossover(p1, p2)
    c1 = mutation(c1)
    c2 = mutation(c2)
    if v:
        print('Child 1:{0}\tFitness: {1}\nChild 2:{2}\tFitness: {3}'.format(c1, fitness(c1,graph), c2, fitness(c2,graph)))
    pop = pop.sort_values(by=['Fitness'])
    pop = pop.reset_index(drop=True)
    pop.iloc[pop.shape[0]-2] = [c1, fitness(c1,graph)]
    pop.iloc[pop.shape[0]-1] = [c2, fitness(c2,graph)]
    return pop
def hasSolution(p):
    if (pop[pop['Fitness']==0]).shape[0] == 0:
        return False
    return True
def bestFitness(pop):
    p = pop.sort_values(by=['Fitness'])
    return p.iloc[0, 1]
def bestChromosome(pop):
    p = pop.sort_values(by=['Fitness'])
    return p.iloc[0, 0]
def getSolution(pop):
    return (pop[pop['Fitness']==0]).iloc[0][0]

show_progress = False
lastgeneration = 0

generationInfo = pd.DataFrame(np.zeros((5000, 2)), columns=['Chromosome', 'Fitness'], dtype=object)

for generation in range(5000):
    lastgeneration = generation
    if generation == 0:
        pop = randinit(500, graph, colour)
        
    generationInfo.loc[generation] = [bestChromosome(pop), bestFitness(pop)]
    if hasSolution(pop):
        print("Found Solution. {:,} Generations generated.".format(generation))
        break
    if(show_progress):
        print("\nGeneration {}".format(generation))
    pop = replaceWorst(pop, graph, show_progress)
    

generationInfo = generationInfo[generationInfo['Chromosome']!=0]  
if(not hasSolution(pop)):
    print("Solution not found program ended.")print(getSolution(pop))

import matplotlib.pyplot as plt
plt.plot(range(generationInfo.shape[0]), list(generationInfo['Fitness']))
plt.show()