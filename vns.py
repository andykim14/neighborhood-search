
import random
import math
#import goto, label
import numpy as np
import matplotlib.pyplot as plt

node = 16
S = node
size = node
route = []
route_m = [0]* 1
route_n = [0]* node
route_l = []
route_r = []
route_a = []
route_b = []
population = []  # list that holds paths
population_size = 30  # max 120 combinations
ps = 0.55
mutate_prob = 0.01   # probability of mutation
#cx_prob = 0.34      # probability of crossover
n_generations = 50    # Maximum iteration
routes_length = [0]*population_size  # list to store the total length of each chromosome
fitness = [0]*population_size        # list to store the total fitness of each chromosome
temp1,temp2 = [0]* node,[0]* node
best_path = math.inf
L = 300  ##### 300 kg highest load capacity of a vehicle
LL = 950

#random.seed(100)


#cities = [0, 1, 2, 3, 4,5,6,7,8,9]
#cities = []       # list that holds basic input nodes
cities = list(range(node))
best_index = [0] * len(cities)
# distance matrix for our 42 cities
distances = [[1000,18,13,8,21,7,15,12,24,28,32,24,22,26,34,42],
             [18,1000,9,10,9,11,19,6,6,10,14,12,18,22,20,24],
             [13,9,1000,7,8,19,27,14,11,15,19,21,27,31,29,28],
             [8,10,7,1000,13,12,20,8,16,20,24,16,21,25,26,34],
             [21,9,8,13,1000,20,28,15,11,7,12,21,27,31,29,22],
             [7,11,19,12,20,1000,8,5,17,21,25,17,15,19,27,35],
             [15,19,27,20,28,8,1000,13,17,21,17,9,7,11,19,27],
             [12,6,14,8,15,5,13,1000,12,16,20,12,12,16,24,32],
             [24,6,11,16,11,17,18,12,1000,4,8,10,16,20,18,17],
             [28,10,15,20,7,21,22,16,4,1000,5,13,20,24,22,14],
             [32,14,19,24,12,25,17,20,8,4,1000,10,16,20,18,10],
             [24,12,21,16,21,17,9,12,10,13,10,1000,6,10,11,18],
             [22,18,27,21,26,15,7,12,16,20,13,6,1000,3,11,19],
             [26,22,31,25,30,19,11,16,20,24,21,10,3,1000,7,15],
             [34,20,29,26,28,27,19,24,18,22,17,11,12,7,1000,8],
             [42,24,28,34,22,35,27,32,17,14,10,18,19,15,8,1000]]
             
### DEPOT to bin
### DISPOSAL 1 to bin
### DISPOSAL 2 to bin
BTD = [[18,8,17,10,16,12,11,6,7,9,13,6,9,13,15,22],
       [11,10,7,4,14,18,24,12,17,20,23,19,25,29,27,34],
       [36,22,30,27,30,28,19,21,20,25,23,14,18,15,4,12]]


# calculates distance between 2 cities
def calc_distance(city1, city2):
    return distances[city1][city2]  # ord('A')=65

def calc_distance1(city1, city2):
    return BTD[city1][city2]  # ord('A')=65

# creates a random route
def create_route():
    shuffled = random.sample(cities, len(cities))
#    print(shuffled)
    return shuffled


# calculates length of an route
def calc_route_length():
    #size = len(cities)
    for i in range(population_size):
        route_l = 0
        for j in range(1,len(cities)):
            route_l = route_l + calc_distance(population[i][j-1], population[i][j])
        route_l = route_l + calc_distance(population[i][size-1], population[i][0]) #calculate distance from last to first
        routes_length[i] = route_l
        fitness[i] = 1 / routes_length[i]
       # print("fitness=", fitness[i])

# creates starting population
def create_population():
    for ik in range(population_size):
        population.append(create_route())


# swap with a probability 2 cities in a route
def swap_mutation():
    picks = random.sample(range(S), 2)
    route_a[picks[0]], route_a[picks[1]] = route_a[picks[1]], route_a[picks[0]]
    
def adapt_path():
    S1 = 0
    min=1000
    route_m = []
    Dist1 = 0 
    Dist2 = 0
    global D1,D2 

    #r1 = random.randint(1,node-1)
        
    ##### Fit the load for each bin
    #L1=0
    #while L1 <= LL :
    L1=0
    route = random.sample(range(30, 92), S)
    for i1 in range(S) :
        L1 += route[i1]

    print("L1 = ",L1)
    '''
    if S == 11 :
       route = [92,83,85,94,96,94,96,87,50,95,96]
    if S == 10 :
       route = [94,93,95,94,96,94,96,96,95,96]
    '''   

    print("Load of bins =",route)
    for i1 in range(S):
        if route[i1] >= 80 :
            route_m.append(i1)
    r1 = len(route_m)
    print("How many bins fill > 80% ?  ",r1)

    #route_m = random.sample(temp1,r1)
    print("The BINs are ",route_m)
    
    ### check the total load of all bins
    l = 0
    for i1 in range(0,len(route_m)) :
        i = route_m[i1]
        l += route[i]
        print("L=",l)
    
    route_m1 = route_m[:]
    
    
    for i1 in range(r1):
        i2 = calc_distance1(0,route_m[i1])
        #print("I2 = ",i2)
        if i2 < min :
           min = i2
           i3 = route_m[i1]
    #print("Min= ",i3)
    S1 = i3
    Min = min
    route_l.append(i3)
    route_r.append(i3)
    
    min = 1000
    for i1 in range(r1):
        i2 = calc_distance1(0,route_m[i1])
        #print("I2 = ",i2)
        if i2 < min and i2 > Min:
           min = i2
           i3 = route_m[i1]
    #print("MINN= ",i3)
    S2 = i3
    
    
    
    k = route_m.index(S1)
    k1 = route_m1.index(S1)
    #print("k = ", k)
    route_m[k] = -1
    route_m1[k1] = -1
    
    #print("2ND = ",route_m)
    #print("2NDDD = ",route_m1)

####### select the left node
    i2,i3=0,0
    
    while all(x == route_m[k] for x in route_m) != True:
        mini = math.inf
        for i2 in range(r1) :
             if route_m[i2] != -1 :   
               v1 = calc_distance(S1,route_m[i2])
               #print("First v1 = ",v1)
               if v1 < mini :
                 mini = v1
                 i3 = route_m[i2]
        #print("Min1= ",i3)
        S1 = i3
        #print("1st L = ", i3)
        k = route_m.index(i3)
        route_m[k] = -1

        route_l.append(i3)
        #print("Left = ",route_l)
        #print("After left one remove =", route_m)
    
########### select the right node
    i2,i3=0,0
    route_r.append(S2)
    k1 = route_m1.index(S2)
    route_m1[k1] = -1
    while all(x == route_m1[k] for x in route_m1) != True:
        mini = math.inf
        for i2 in range(r1) :
             if route_m1[i2] != -1 :   
               v1 = calc_distance(S1,route_m1[i2])
               #print("First v1 = ",v1)
               if v1 < mini :
                 mini = v1
                 i3 = route_m1[i2]
        #print("Min1= ",i3)
        S1 = i3
        #print("1st L = ", i3)
        k = route_m1.index(i3)
        route_m1[k] = -1

        route_r.append(i3)
        #print("Right = ",route_r)
        #print("After right one remove =", route_m1)
    
    ####### create tree structure
   

        #min_dist = math.inf
      ##################################################
    
    for i1 in range(1,len(route_l)) :
        Dist1 += calc_distance(route_l[i1-1],route_l[i1])
    Dist1 += Min
    for i1 in range(1,len(route_r)) :
        Dist2 += calc_distance(route_r[i1-1],route_r[i1])
    Dist2 += Min
    
    #print("Dist1 =", Dist1)
    #print("Dist2 =", Dist2)
    if BTD[1][route_l[-1]] < BTD[2][route_l[-1]] :
        i2 = BTD[1][route_l[-1]]
        Disposal = 1
    else :
        i2 = BTD[2][route_l[-1]]
        Disposal = 2
    
    Dist1 += i2
    print("Final Distance1 =",Dist1)
    print("Path from starting Depot to",route_l)
    #print("I2 = ",i2)
    print("and Disposal Center = ",Disposal)

    if BTD[1][route_r[-1]] < BTD[2][route_r[-1]] :
        i3 = BTD[1][route_r[-1]]
        Disposal = 1
    else :
        i3 = BTD[2][route_r[-1]]
        Disposal = 2
        
    Dist2 += i3
    print("Final Distance2 =",Dist2)
    print("Path from starting Depot to",route_r)
            
    #print("I3 = ",i3)
    print("and Disposal Center = ",Disposal)
    D1,D2 = Dist1,Dist2
    
    ####  Vehicle selection####  How many vehicles
    
    if l > L :
        n_v = (l /L)
        if (l %L) > 0 :
            n_v += 1
    else:
        n_v = 1
    print("Number of vehicle required =", n_v)    
 


def bin_aloc():
    
    S11 = 0
    min=1000
    global D11,D22
    
    for i1 in range(S):
        route_n[i1]=i1
     
    route_n1 = route_n[:]
    route_n2 = route_n[:]
    min = 10000    
    for i1 in range(S):
        i2 = calc_distance1(0,route_n[i1])
        #print("I2 = ",i2)
        if i2 < min :
           min = i2
           i3 = route_n[i1]
    #print("Min= ",i3)
    S11 = i3
    Min1 = min
    route_a.append(i3)
    #i3 = random.randint(0,S-1)
    route_b.append(i3)
    
    min = 10000
    for i1 in range(S):
        i2 = calc_distance1(0,route_n[i1])
        #print("I2 = ",i2)
        if i2 < min and i2 > Min1:
           min = i2
           i3 = route_n[i1]
    #print("MINN= ",i3)
    S12 = i3
    
    
    
    k2 = route_n1.index(S11)
    k3 = route_n2.index(S11)
    #print("k = ", k)
    route_n1[k2] = -1
    route_n2[k3] = -1
    
    #print("2ND = ",route_m)
    #print("2NDDD = ",route_m1)

####### select the left node
    i2,i3=0,0
    
    while all(x == route_n1[k2] for x in route_n1) != True:
        mini = math.inf
        for i2 in range(S) :
             if route_n1[i2] != -1 :   
               v1 = calc_distance(S11,route_n1[i2])
               #print("First v1 = ",v1)
               if v1 < mini :
                 mini = v1
                 i3 = route_n1[i2]
        #print("Min1= ",i3)
        S11 = i3
        #print("1st L = ", i3)
        k2 = route_n1.index(i3)
        route_n1[k2] = -1

        route_a.append(i3)
        #print("Left = ",route_l)
        #print("After left one remove =", route_m)
    
########### select the right node
    i2,i3=0,0
    route_b.append(S12)
    k3 = route_n2.index(S12)
    route_n2[k3] = -1
    while all(x == route_n2[k3] for x in route_n2) != True:
        mini = math.inf
        for i2 in range(S) :
             if route_n2[i2] != -1 :   
               v1 = calc_distance(S11,route_n2[i2])
               #print("First v1 = ",v1)
               if v1 < mini :
                 mini = v1
                 i3 = route_n2[i2]
        #print("Min1= ",i3)
        S11 = i3
        #print("1st L = ", i3)
        k2 = route_n2.index(i3)
        route_n2[k2] = -1

        route_b.append(i3)
        #print("Right = ",route_r)
        #print("After right one remove =", route_m1)
    
    ####### create tree structure
   

        #min_dist = math.inf
      ##################################################
    Dist11 = 0
    Dist22 = 0
    
    for i1 in range(1,len(route_a)) :
        Dist11 += calc_distance(route_a[i1-1],route_a[i1])
    Dist11 += Min1
    for i1 in range(1,len(route_b)) :
        Dist22 += calc_distance(route_b[i1-1],route_b[i1])
    Dist22 += Min1
    
    D11,D22 = Dist11,Dist22
    print("Alocation-1, Alocation-2",Dist11,Dist22)
    
    print("Least BIN alocation path-a: ",route_a)
    print("Least BIN alocation path-b: ",route_b)
    
    
#    swap_mutation()

#    for i1 in range(1,len(route_a)) :
#        Dist11 += calc_distance(route_a[i1-1],route_a[i1])
#    Dist11 += Min1
#    for i1 in range(1,len(route_b)) :
#        Dist22 += calc_distance(route_b[i1-1],route_b[i1])
#    Dist22 += Min1
    
#    print("Alocation-1, Alocation-2",Dist11,Dist22)
    



    y = list(range(node))

    x = route_a
    #y = route_b


# plotting points as a scatter plot 
    plt.scatter(x, y, label= "stars", color= "green", linestyle='dashed', linewidth = 1,  marker= "o", s=10)

# plotting the points  
    #plt.plot(x, y, color='black', linestyle='dashed', linewidth = 1, marker='o', markerfacecolor='yellow', markersize=3)
    plt.figure(figsize=(12,6)) 
  
    ax = plt.axes() 
    ax.grid(linewidth=0.1, color='#8f8f8f')
#ax.set_facecolor("black") 
    plt.plot(x, y, color='#1F77A4', 
        marker='o', 
        linewidth=1, 
        markersize=5, 
        markeredgecolor='#035E9B')
  
# setting x and y axis range 
    plt.ylim(0,node+1) 
    plt.xlim(0,node+1) 
  
# naming the x axis 
    plt.xlabel('Alocation path') 
# naming the y axis 
    plt.ylabel('Bin') 
  
# giving a title to my graph 
    plt.title('All BINs Alocation') 
  
# function to show the plot 
    #plt.show() 


    
bin_aloc()    

adapt_path()
if D1 < D2 :
    Dist = D1
else:
    Dist = D2
print(Dist)
print(D11)
Tcost = Dist + D11
print("Total cost = ",Tcost)

