import itertools
import time
import pandas as pd
import numpy as np


#pierwsza formuła   50zmiennych, 218 klauzul
x=pd.read_csv("uf20-01.cnf", sep=" ", header=None)
arr=x[:-2].to_numpy()
arr=arr.tolist()

formula=[]
for i in range(len(arr)):
    a=[int(x) for x in arr[i][:-1] ]
    formula.append(a)



variations=list(itertools.product([1,0],repeat=20))

def brute_force(formula):
    for variation in variations:
        
        sum=0
        for claus in range(len(formula)):
            jedynki=0
            for element in range(3):
                value=variation[abs(formula[claus][element])-1]
                
                if formula[claus][element]<0:
                    if value==0:
                        value=1
                        jedynki+= 1
                elif formula[claus][element]>0:
                    if value==1:
                        jedynki+=1

                if jedynki>0:
                    sum+=1
                    break
        if sum==91:
            print(sum, variation)
            break

start_time=time.time()

print(brute_force(formula))

print(time.time()-start_time)






