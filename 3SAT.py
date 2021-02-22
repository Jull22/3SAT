from pyeasyga import GeneticAlgorithm
from statistics import mean
import time
import matplotlib.pyplot as plt



class Test(GeneticAlgorithm):

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
    the supplied fitness_function.
    """
        fitness_in_generation=[]
        

        for individual in self.current_generation:
            
            individual.fitness = self.fitness_function(
                individual.genes, self.seed_data)
            fitness_in_generation.append(individual.fitness)

        max_fitness_in_generation.append(max(fitness_in_generation))
        mean_fitness_in_generation.append(mean(fitness_in_generation))

        
    
    def run(self):
        """Run (solve) the Genetic Algorithm."""
        
        self.create_first_generation()

        for _ in range(1, self.generations):
            self.create_next_generation()
        # print(max_fitness_in_generation)
        # print(mean_fitness_in_generation)




import pandas as pd
import numpy as np


#pierwsza formuła   20zmiennych, 91 klauzul
x=pd.read_csv("uf20-01.cnf", sep=" ", header=None)
arr=x[:-2].to_numpy()
arr=arr.tolist()

formula=[]
for i in range(len(arr)):
    a=[int(x) for x in arr[i][:-1] ]
    formula.append(a)
# print(formula)


#druga formuła    50 zmiennych, 218 klauzul
data2=pd.read_csv("uf50-030.cnf", sep=" ", header=None)
arr2=data2[:-2].to_numpy()
arr2=arr2.tolist()

formula2=[]
for i in range(len(arr2)):
    a=[int(x) for x in arr2[i][:-1] ]
    formula2.append(a)

#trzecia formuła    150 zmiennych, 645 klauzul
data3=pd.read_csv("uf150-01.cnf", sep=" ", header=None)
arr3=data3[:-2].to_numpy()
arr3=arr3.tolist()

formula3=[]
for i in range(len(arr3)):
    a=[int(x) for x in arr3[i][:-1] ]
    formula3.append(a)

def fitness(individual, formula):
    sum=0
    for claus in range(len(formula)):
        for element in range(3):

            value=individual[abs(formula[claus][element])-1]
            
            if formula[claus][element]<0:
                if value==0:
                    value=1
                    sum+=1
                    break
            elif formula[claus][element]>0:
                if value==1:
                    sum+=1
                    break

    return sum



formulas=eval(input("Wprowadź formułę: ('formula', 'formula2' lub 'formula3'): "))
population=int(input("Wprowadź wielkość populacji: "))
generations=int(input("Wprowadź ilość generacji: "))
mutacja=float(input("Wprowadź prawdopodobieństwo mutacji: "))
elityzm=input("Elityzm True/False: ")


ga = Test(formulas, population_size=population,
                        generations=generations,
                        mutation_probability=mutacja,
                        elitism=elityzm)


ga.fitness_function = fitness               # set the GA's fitness function
max_fitness_in_generation=[]
mean_fitness_in_generation=[]

start_time=time.time()
ga.run()
 
print(time.time()-start_time)

print(ga.best_individual())                            


import matplotlib.pyplot as plt

fig, ax = plt.subplots() 
ax.plot(max_fitness_in_generation, '-r',label='maksymalna')
ax.plot(mean_fitness_in_generation, '-b', label='średnia')
ax.legend()
plt.title("Działanie algorytmu genetycznego")
plt.ylabel('ocena fitness')
plt.xlabel('pokolenia')
plt.show()


#Przykład dla losowego chromosomu (9zmiennych)

# chromosom=[0,1,1,1,0,1,1,0,1]
# example_formula=[[-2,1,3],[4,5,-6],[-7,-9,8],[-1,4,-5]]
# print(fitness(chromosom,example_formula))


# fig, ax = plt.subplots() 
# plt.plot(["GA input=91", "GA input=218", "GA input=645","brute force(input=91)"],[0.85, 24, 228, 27.93], color="r")
# plt.bar(["GA input=91", "GA input=218", "GA input=645","brute force(input=91)"],[0.85, 24, 228, 27.93])
# plt.title("Czas znalezienia rozwiązania")
# plt.ylabel('czas działania algorytmu w sekundach')
# plt.xlabel('wielkości inputów')
# plt.show()