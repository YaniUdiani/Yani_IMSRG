#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:12:26 2020

@author: YaniUdiani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:31:38 2020

@author: YaniUdiani
"""
#------------------------------------------------------------------------------
""" 
This version is based of Adaption_dev8
"""
#------------------------------------------------------------------------------

import numpy as np
from numpy import dot, reshape, transpose
#from scipy.integrate import odeint, ode
from numpy import linalg as LA # For Frobenius norm
from fractions import Fraction as Fr # For Bernoulli Numbers
#from sys import argv
import sys
import matplotlib.pyplot as plt # Plotting stuff
import copy
import time
#import numba
#from numba import jit
import itertools as tools #for infinite matter states
import cProfile
from collections import Counter
#-----------------------------------------------------------------------------------
# Ploting variables
#-----------------------------------------------------------------------------------
flow=[] #list of flow parameters
energy=[]#list of energies for plotting
eigs=[]
HH=[]
initializations = [] #used to store E_ref, and f_Hartree-fock
results = [] #used to store E_g.s, f,Gamma, Omega1B, Omega2B 
global_user_data =[]
bass = []
bass1 = []
full = []
fully = []
countering = []
#-----------------------------------------------------------------------------------
# Functions to compute Infinite Matter single particle states
#-----------------------------------------------------------------------------------
def powerset(iterable): #Function from itertools to find powerset
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return tools.chain.from_iterable(tools.combinations(s, r) for r in range(len(s)+1))

# The full_state generator below works by guessing and checking all microstates below the N_Max. 
# Below the N_Max is determined by <=sqrt(N_Max). Any state with a component of momenta greater than sqrt(N_Max) will be above N_Max.
# These fuctions only return "positive states": states with orbitals that have positive momenta
# Spin projections are added at the end of every microstate
# An example of a state is [0,0,0,0.5] 
def full_state_gen_3D(N_Max, degen): #3D "positive state" generator
  
    max_k_comp=np.floor(np.sqrt(N_Max))#define maximum wavenumber that won't exceed N_Max
    statey=[]
    if(degen == 2):#if you're in an isospin triplet (pure neutron matter for ex)
      
        for i in range(-int(max_k_comp),int(max_k_comp+1)):#int used ensure proper data type

            for j in range(-int(max_k_comp),int(max_k_comp+1)):
              
                for k in range(-int(max_k_comp),int(max_k_comp+1)):#+1 to ensure that the max k is included
                  
                    if((i**2 + j**2 + k**2)<=N_Max):#check to make sure it is at or below the N_Max
                      
                        for spin in [1/2,-1/2]:
                          
                            statey.append([i, j, k, spin])
    
    if(degen == 4): #if you have both protons and neutrons
      
        for i in range(-int(max_k_comp),int(max_k_comp+1)):#int used ensure proper data type
          
            for j in range(-int(max_k_comp),int(max_k_comp+1)):
              
                for k in range(-int(max_k_comp),int(max_k_comp+1)):#+1 to ensure that the max k is included
                  
                    if((i**2 + j**2 + k**2)<=N_Max):#check to make sure it is at or below the N_Max
                      
                        for spin in [1/2,-1/2]:
                            
                          for isospin in [1/2,-1/2]:
    
                              statey.append([i, j, k, spin, isospin])
    return statey#returns list of acceptable states

#This function takes in "positive states" and returns the "negative states" corresponding to that state. 
#It also returns the postive state. For example, feeding in [0,1,0.5] will return [0,-1,0.5], [0,-1,-0.5],  [0,1,0.5]  and [0,1,-0.5]
#This is done to easily include the positive states and their negative equivalents in full_state variable below
def sign_gen(inputy):
    array=list(range(0,len(inputy)))
    results=list(powerset(array))
    all_sign_permutations=[]
    all_sign_permutations.append(inputy)
    for k in  range(1,len(results)):
        dummy_list=inputy[:]# Copies inputy to dummy_list
        if 0 not in [inputy[i] for i in results[k]]:#[inputy[i] for i in results[k]]=inputy[results[k]] in R speak
        ### input arrays with zeros will double count states, this condition takes care of that ####
            for j in results[k]:
                dummy_list[j]=-inputy[j]# flip signs
            all_sign_permutations.append(dummy_list)
            #print(dummy_list)
    return all_sign_permutations

#This function sorts states so that holes come before particles in Full_state. I prefer it this way. 
def sorter(temp_full_state, Fermi_lvl):
    holes=[]
    particles=[]
    for i in temp_full_state:
        if((i[0]**2+i[1]**2+i[2]**2)<=abs(Fermi_lvl)):
            holes.append(list(i))
        else:
            particles.append(list(i))
    return holes+particles

#The function below generates the particles and hole states that we're used to
#The function makes my infinite matter generator easily adaptable to my Magnus code
#This function assumes that the box is 3-D
def One_BD_States(Fermi_lvl, full_state): 
    particles=[]
# This loop will decrement from end of full_state to beginning, why? Cuz, the particles will be at the end!
    for i in range(len(full_state)-1, -1, -1): #range goes from len(full_state)-1 to 0 by units of -1
        if ((full_state[i][0]**2+full_state[i][1]**2+full_state[i][2]**2)> abs(Fermi_lvl)):#Note that this Fermi_lvl is the true Fermi_lvl
            #If it's a particle, append the index to the particle list
            #abs(Fermi_lvl) chosen to allow fermi lvl=-1 for easily testing some of my functions
            particles.append(i)
        else:#The moment it is not a particle, break. I'll manually tell it the indices that are holes.
            break
    holes=list(range(0, i+1))#holes are locations from 0 to i since i is below the fermi_lvl
    return holes, list(reversed(particles)) #reverse particles to ascending order because I looped backwards!

#-----------------------------------------------------------------------------------
# basis and index functions
#-----------------------------------------------------------------------------------
#Generates all possible pairs of single particle states and orders them such that the pairs that have the same
#sums of momenta are grouped together. It returns bas2B implicity organized in terms of blocks,
#it returns block_sizes which is an array of sizes of each block- generally useful
#it returns bas_block2B which organizes bas2B explicity in terms of blocks- will be useful for OP_Map class
#it returns Subset which is useful for constructing the Hamiltonian:
    
#For every pair (call it p) in Bas2B, the use of anti-symmetrized matrix elements allows me to calculate matrix elements of that pair
#with any pair in Bas2B, in addiiton to allowing me to simulatenously calculate matrix elements of pairs (call them g) in the permuation
#group of p with any pair in Bas2B. See lab notebook for detailed examples. Therefore, I need to loop over "unique" pairs in Bas2B
#when I'm calculating my 2-BD Hamiltonian. By "unique", I mean that I want one pair from each disjoint permuation group in Bas2B. This will save me on
#much needed compuation time. Also, I'm not going to consider terms like (r,r) in my matrix element so might as well select them out here as well. 
# The list subset2B is a filter of bas2B to including the unique pairs.

def construct_basis_2B(full_state,states):

    CM_Momenta=[]#list of all CM momenta
    pairs=[]
    block={}
    Bas2B=[]#gives all possible pairs grouped by CM momenta
    block_sizes=[]#number of pairs in a given block of Hamiltonian
    subset_sizes=[]#number of pairs in Subset
    
    for p in states:
        for q in states:
            CM_Momenta.append(full_state[p][0:3]+full_state[q][0:3])
            pairs.append((p,q))#store corresponding pairs for CM momenta
    
    for i, p in enumerate(CM_Momenta):
        if(str(p) not in block): #str(p) are keys corresponding to CM momenta blocks
            block[str(p)]=[pairs[i]]#Create a new block if it doesn't already exist
        else:
            if(pairs[i][0] <= pairs[i][1]): #store upper triangle first
                block[str(p)].append(pairs[i])#store pairs in a given block     
   
    #print(block)
    
    block = Symmetries_of_Hamiltonian(full_state, block) #do more filtering on block
    #print(block)
    num_blocks=len(block)#get number of blocks
    
    mom_to_blk_nums = {}# keys as momenta+spin, values = block_num
    Subset=[]# List will contain lists of filterd pairs in a given block.
    #This list filters the pairs in each block. It acts as subset2B for each block    
    Bas_block2B=[[] for i in range(num_blocks)]# Bas2B with each block clearly demarcated
    block_num = 0
    
    for keys in block:
      
        mom_to_blk_nums[keys] = block_num
        #Remember that block[keys] is upper triangle including diag elements
        same_ij = [(i,j) for (i,j) in block[keys] if i==j]
        #same_ij stores (i,i) terms in block[keys] so that they'll be stuck at 
        #the very end of the block. That way, they don't interfere with my
        #mapping. I want to build my basis such that if I know the location of 
        #(i,j) for i!=j, then I know where (j,i) is. This is super useful!
        if same_ij: #if same_ij isn't empty
            blockness = [(i,j) for (i,j) in block[keys] if i!=j]#get terms without (i,i)
            Subset.append(blockness)#store upper triangle without diag (i,i)
            lower_T = [(i,j)[::-1] for (i,j) in blockness]#flip pairs
            block[keys] = blockness + lower_T + same_ij
        else:
            Subset.append(block[keys])#store upper triangle without diag (i,i)
            lower_T = [(i,j)[::-1] for (i,j) in block[keys]]#flip pairs
            block[keys] = block[keys] + lower_T
            
        for elements in block[keys]:
            Bas2B.append(elements) #get full Bas2B
            Bas_block2B[block_num].append(elements) #store pairs for blocks seperately
            
        #print(Subset[block_num])
        #print(block[keys])
        #print("-------")     
        #dict_version_of_blks[keys] = Bas_block2B[block_num]
        block_sizes.append(len(block[keys]))#Number of pairs in each block
        block_num+= 1
                
    summ=0#test to ensure that I got the right number of pairs in Subset
    
    for k in range(len(Subset)):
        lenny=len(Subset[k])
        subset_sizes.append(lenny)
        summ+=lenny
    
    if((len(Bas2B)-len(Bas2B)**(.5))/2!=summ): #ensure that Subset represents upper triangular matrix of pairs
          print("Something is wrong with construct_basis_2B()")
            
    #block.clear() #is this useful in decreasing memory footprint of block? I dunno
    return Bas2B, Subset, block_sizes, subset_sizes, Bas_block2B, mom_to_blk_nums


def Symmetries_of_Hamiltonian(full_state, block_dict):
  
  """ This function filters the already established momentum blocks into smaller
  blocks based on extra symmetres of the Hamiltonian. For example, 
  since the Minnesota potential conserves total spin projection,
  all operators eta, and omega should conserve it as well. I'm not considering
  iso-spin right now, but likewise is the same for iso-spin. I'll come back and
  update this if this filtering process turns out to give decent performance boosts
  and I'm actually working with symmetric neutron matter. 
  
  block_dict is a dictionary containing pairs (i,j) for j>i in a given 
  momentum block. full_state is a list containing the actual momenta of the 
  single particle states.
  
  Note that I constructed full_state such that if sp label is even, the spin
  projection is +1/2 and if sp label is odd, the spin proj is -1/2.
  This is only true when iso-spin is turned off.
  
  """
  
  new_dict = {} #dict with more filtering
  
  #print(block_dict['[0. 0. 0.]'])
  #print('break')
  
  for key in block_dict:
    
    plus_plus = []
    mixed = []
    minus_minus = []
    
    for (i,j) in block_dict[key]: 
      
      if(i%2 == 0 and j%2 == 0 ):#store spin pairs (1/2,1/2)
        plus_plus.append((i,j))
      
      if(i%2 == 1 and j%2 == 0 ):#store spin pairs (-1/2,1/2)
        mixed.append((i,j))
        
      if(i%2 == 0 and j%2 == 1 ):#store spin pairs (1/2,-1/2)
        mixed.append((i,j))     
        
      if(i%2 == 1 and j%2 == 1 ):#store spin pairs (-1/2,-1/2)
        minus_minus.append((i,j))

   
    if(plus_plus):#if full, create new block with momentum key & total spin 1
      new_dict[key + "0"] = plus_plus
      
    if(mixed):#if full, create new block with momentum key & total spin 0
      new_dict[key + "1"] = mixed

    if(minus_minus):#if full, create new block with momentum key & total spin -1
      new_dict[key + "2"] = minus_minus
      
  return new_dict
  
#Modifies bas_block2B to return all specific pairs that will be required by specific loops in
#commutator routine, and also eta construction
def pair_selector(bas_block2B, particles, holes):
    lenny = range(len(bas_block2B))
    Particle_Particle = [[] for i in lenny]
    Hole_Hole = [[] for i in lenny]
    Particle_Anything = [[] for i in lenny] #sperate P_H, and P_P pairs
    Hole_Anything = [[] for i in lenny]

    for block_num in lenny:
        for pairs in bas_block2B[block_num]:
            if(pairs[0] in particles):
                Particle_Anything[block_num].append(pairs)
                
                if(pairs[1] in particles and pairs[1] > pairs[0]):
                    Particle_Particle[block_num].append(pairs)
            else: #pairs[0] is in holes
                Hole_Anything[block_num].append(pairs)
                
                if(pairs[1] in holes and pairs[1] > pairs[0]):
                    Hole_Hole[block_num].append(pairs)      
                    
    return Particle_Particle, Hole_Hole, Particle_Anything, Hole_Anything 
  
  
def ph_filtering(Hole_H, Particle_P, Hole_A):
  
  """ This function returns block numbers in the inputs that are not empty"""
  HH_PP_Filter = []
  
  Hole_A_Filter = []
  
  Lubos = len(Hole_H)
  
  counter = 0
  counter1 = 0
  
  for block_num in range(Lubos):
    
    if(Particle_P[block_num] and Hole_H[block_num]): #If they both have elements
      
      HH_PP_Filter.append(block_num)
      
      counter += 1
      
    if(Hole_A[block_num]):
      
      Hole_A_Filter.append(block_num)
      
      counter1 += 1
      
  
  print("Fraction of blks that both have Hole_H and Particle_Hole terms: ", round(counter/Lubos, 4) )
  print("Fraction of blks that both have Hole_Anything pairs: ", round(counter1/Lubos, 4) )
  
  return HH_PP_Filter, Hole_A_Filter    

#This function is used to speed up the 2B term in the 1B-2B comm
#For every block, and every term that is added to Output[(i,j), (k,l)]
#it determines which values of "a" will contribute
#left_connections corresponds to values of "a" where "a" is on the left most side
#of the pair : see the commutator eqs. Likewise for right connections 
#Note: I might change the labeling of "a" to something like "p" in the future
#to be consistent with chem notation.
def subset_selector(subset2B):
    
  left_connections=[{} for i in range(len(subset2B))]# contains sp states for looping for each blk
  right_connections=[{} for i in range(len(subset2B))]# contains sp states for looping for each blk
  
  for block_num, blocks in enumerate(subset2B):
    
    for (i,j) in blocks:
        
      if (i,j) not in left_connections[block_num].keys():#create lists in dict
        left_connections[block_num][(i,j)] = []
        right_connections[block_num][(i,j)] = []
        
      for (k,l) in blocks:
        if(j == l):
          left_connections[block_num][(i,j)].append(k)#corresponds to the a in states that is non-zero         
        if(i == k):
          right_connections[block_num][(i,j)].append(l)#corresponds to the a in states that is non-zero   
         
  return left_connections, right_connections

             
class OP_Map:#class maps to a given operator allowing calls to that operator's elements. These OPs are of the form [[]]
#This class is super useful because it allows me to keep a similar indexing structure of a single matrix even though I'm
#Using a list of block matrices for my operators
  
    #Initialize class vars to None. They'll be changed once the vars are known.
    obj = None #Each instance of OP_Map_B is a list operator, so obj will be 
    #routinely replaced locally by each instance: each instance will see its operator.
    idp = None #idp is shared across all instances of OP_Map_B, so it is a fixed
    #class var: it won't be changed locally or globally
    #idp_b = None #idp is shared across all instances of OP_Map_B, so it is a fixed
    #class var: it won't be changed locally or globally
    def __init__(self, tag):#use operator and idp to define self since those are universally
        self.tag = tag
        

    def __getitem__(self, pairs): # Will activate if you do a simple call on OP_Map
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] != second[0] ):#if they aren't in the same block
            return 0.0
        else:            
            return self.obj[first[0]][first[1], second[1]]
        
    def __setitem__(self, pairs, val):# Activates when OP_Map()=val
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] == second[0] ):#if they are in the same block
            self.obj[first[0]][first[1], second[1]] = val
            
    def __add__(self, pairs, val):# Activates when OP_Map()+=val
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] == second[0] ):#if they are  in the same block
            self.obj[first[0]][first[1], second[1]] += val
        
    def __sub__(self, pairs, val):# Activates when OP_Map()-=val
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] == second[0] ):#if they are in the same block
            self.obj[first[0]][first[1], second[1]] -= val   
   
class OP_Map_B:# Does same thing as OP_Map, but if the blk_num of both 
  #pairs is known, then it will search that block for their indices
  
  #Initialize class vars to None. They'll be changed once the vars are known.
  obj = None #Each instance of OP_Map_B is a list operator, so obj will be 
  #routinely replaced locally by each instance: each instance will see its operator.
  idp_b = None #idp_b is shared across all instances of OP_Map_B, so it is a fixed
  #class var: it won't be changed locally or globally
  
  def __init__(self, tag):
    self.tag = tag #dummy var: instance referenced through tag
    
  def __getitem__(self, pairs): # Will activate if you do a simple call on OP_Map_B
    #print(self.obj) 
    return self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]]
        
  def __setitem__(self, pairs, val):# Activates when OP_Map_B()=val
    self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]] = val
              
  def __add__(self, pairs, val):# Activates when OP_Map_B()+=val
    self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]] += val
    
  def __sub__(self, pairs, val):# Activates when OP_Map_B()-=val
    self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]] -= val
      
#Quick test to ensure that generic OP_Map does what it needs to do            
def OP_Map_Test(bas2B, block_sizes, bas_block2B, idp , H2B):
    
    H2B_Test = copy.deepcopy(H2B)
    block_num = block_sizes.index(min(block_sizes))#pick first smallest block
    block = bas_block2B[block_num]
    
    H2B_Test_Map = OP_Map("H2B_Test_Map")
    H2B_Test_Map.obj = H2B_Test
    
    #block=random.choice(bas_block2B)#pick random block (need to import random to work)
    #block_num=bas_block2B.index(block)
    for bra in block:
        for ket in block:           
            prev=H2B_Test[block_num][block.index(bra),block.index(ket)]
            
            ######## Can it call the appropriate matrix elements? ##########
            if(prev != H2B_Test_Map[[ bra, ket ]]):
                print("OP_Map fails at calling H2B(bra,ket) :",bra,ket,block_num)
                sys.exit()
            
            ######## Can it add/sub to matrix elements? ##########
            H2B_Test_Map[[ bra, ket ]]+=2 
            if(H2B_Test[block_num][block.index(bra),block.index(ket)] != prev + 2):
                print("OP_Map fails at adding to H2B(bra,ket) :",bra,ket,block_num)
                sys.exit()

                
            ######## Can it replace matrix elements? ##########
            H2B_Test_Map[[ bra, ket ]]=4            
            if(H2B_Test[block_num][block.index(bra),block.index(ket)] != 4):
                print("OP_Map fails at replacing H2B(bra,ket) :",bra,ket,block_num)
                sys.exit()

    return "Yaaay! OP_Map works as expected :-]"
    
#
# We use dictionaries for the reverse lookup of state indices
#
def construct_index_2B(bas2B):
  index = { }
  for i, state in enumerate(bas2B):
    index[state] = i

  return index

#The next function below does the same thing as idx2B, but stores the indices of pairs in a given block
#Figures out which block in which a given pair (p,q) lies. It also returns the index location of that 
#pair in bas_block2B: that is the row/column # of that given pair in Operator[block]
def special_index(bas_block2B):
  index = {} #dict containing pairs as keys to pair index -- not blocked
  index_blocked = {} #dict containing block_nums as keys to dicts wtih pairs as keys to pair index
  for block_num, bas in enumerate(bas_block2B):
    index_blocked[block_num] = {} 
    for pair in bas:
      ind = bas.index(pair)
      index[pair] = (block_num, ind)
      index_blocked[block_num][pair] = ind
      
  return index, index_blocked

#This function is for the 2BD terms in the commutators that use subset2B[block_num]
#This function reverses subset2B[block_num] in the enumeration, but still keeps the 
#initial indices for the pairs in subset2B[block_num]. This is because
#I want to do the loop over subset2B[block_num] backwards since I'll be deleting
#(i,j) from Hermitian- I don't want the indices of other pairs in Hermitian
#to change when I delete (i,j), so I start collecting (i,j) from the end of 
#subset2B[block_num]
def reverse_enumerate(subset2B_blk,length): 
    return zip(list(range(length-1, -1, -1)), reversed(subset2B_blk))
#-----------------------------------------------------------------------------------
# transform matrices to particle-hole representation
#-----------------------------------------------------------------------------------
    
def extra_symm_of_Ham(a, b):
  
  """ A momentum blk of the ph transformed
  Hamiltonian is discovered, i.e. B = {(a,b),(c,d),(e,f)..., st. k_a-k_b = K} where
  K is some momentum vector with norm ranging from [0,0,0] to \vec{N}. Where N is in the 
  highest particle state. \vec{N} looks like [0,0,N] or [0,N,0] or [N,0,0].
  
  I want to further split B into smaller blocks so that each new block only contains pairs that
  have the same total spin projection if I were to inverse transform back to the
  standard Hamiltonian.
  
  Remember that I organize the sp labels to be even -> spin up
  and odd -> spin down. Since the Minnesota potential preserves total spin projection,
  I want to ensure that 
    
  1.) if (a,b) is even-even, then (c,d) is either even-even or odd-odd
  
  2.) if (a,b) is even-odd, then (c,d) is even-odd
  
  3.) if (a,b) is odd-even, then (c,d) is odd-even
  
  Quick note that the PH trans preserves hermiticity.
  
  This is true because to get the Hamiltonian (H) in standard representation, 
  
  H_ph[ (a,b), (c,d) ] = -H_std[ (a,d), (c,b) ]   :-]
  
  This function uses these facts to obtain the blocks of H_ph
  
  """
  return a%2 - b%2 #0 for odd-odd/even-even, 1 for odd-even, -1 for even-odd
  #This sign identifies the 3 blocks that a given momentum block of the ph_trans
  #will be split into
  

def pair_acceptor(a, b, c, d):
  
  """ph mat elements of form (a,a)(b,b) will be zeroed out by occ_matrix.
  These pairs (a,a) will have k_a-k_a=[0. 0. 0.]
  Moreover, (a,a) won't connect with any other pairs except (b,b).
  Moreover, some pairs of the form (a,a+1) or (a+1,a) will have like (0,1) or (2,3) or (1,0) or (3,2)
  k_{a}-k_{a+1}=[0. 0. 0.] since they will spin up, spin down pairs with same k = k_a
  The pairs (a,a) and (a,a+1) will be the only pairs that form the k_{a}-k_{b}=[0. 0. 0.]
  block. Stare at full_state to clearly see this. 
  Please note that a pair like (8,7) or (7,8) won't fall into this category."""
  
  """I'm keeping this here, so I can later analyze its 
  #Performance with Cprofile. Am I really better checking all these conditions
  #for every pair verus just computing the zero block with smaller number of
  #conditionals (see Dev4), then throwing it out?"""
  
  if (a%2 + d%2 == c%2 + b%2 and d!=a!=b!=c ): #first: conserve total spin projections
    #The a!=b piece is to avoid pairs of the form (a,a). d!=a is manually coding in
    #exlusion principle, likewise for b!=c
    
    if (abs(a-b) == 1 and (1 - (a+b))%4 == 0): #Then you have an element in the special set
        #i.e., you're considering something in the [0,0,0] block. I don't want this blk
        #so return False. If I had another pair with abs(a-b) == 1 in another block
        #it would return True. Note that in the [0,0,0]  blk, if abs(a-b) == 1, then
        #abs(c-d) == 1 since I'm excluding the (a=b) and (c=d) terms. 
        #Note if (a=b), then (c=d) if spin projections are preserved. 
        
        return False
      
    else:
      
      return True
             
  return False

def obtain_blocks(ph_blocks, dim1B, dim2B, fermi_index):
  
  "Returns blocks as list, and orders each block so that ph terms come first"

  basph_blocks2B = []
  
  stats = 0 #ensure that size of blocks is len(bas2B)-len(bas1B)
  stats1 = 0 #total size of ph blocks that will be compared to size of 2B matrix 
  #in the naive case
  
  del ph_blocks['[0. 0. 0.]-1'] #remove '[0. 0. 0.]-1' block since it won't have an ph terms
  #since '[0. 0. 0.]' block will look like (0,1)(2,3)(4,5)... it will have signature = -1
  
  for blockheads in ph_blocks:
    
    ph_arrangement = [] #arrange them so that ph/hp terms come first
    
    temp = []#place holder for pp/hh terms
    
    blockys = ph_blocks[blockheads]
    
    stats += 2 * len(blockys) #multiply be 2 for mirrored blks not included
    #bugs[blockheads] = len(blockys) 
    
    for (i,j) in blockys:
      
      #if(i < fermi_index and j >= fermi_index): #hole-particle term
      if(i < fermi_index <= j or j < fermi_index <= i): #hole-particle and particle-hole terms  
        
        ph_arrangement.append((i,j))
        
      else:
        
        temp.append((i,j))
       
    #print(ph_arrangement+temp) 
    
    if(ph_arrangement):#there has to be at least one ph term for block not to be zeroed out
      #if there are no ph terms in a block, then I don't want to store it 
      basph_blocks2B.append(ph_arrangement + temp)
      
      stats1 += 2*len(blockys)**2 #multiply be 2 for mirrored blks not included
      
#      if(temp):
#        print("I found a block with p-p or h-h terms!")
#        print(ph_arrangement + temp) 
        
  print("Total size of ph blocks compared to size of full 2B matrix: ", round(stats1/(dim2B**2), 6) )
  
  #print(bugs)  
  #print(stats)
  if(stats != dim2B - 2 * dim1B ): #2*len(full_state)= len(full_state)+ len(full_state)
    #with len(full_state) representing #of (a,a) pairs, and len(full_state) representing
    #of (a,a+1) and (a+1,a) pairs representing (same k with some spin, same k with opposite spin) pairs
      
    print("The ph block algorithm has failed to include some pairs in blocks that they should be")
    print("Those pairs are not of the form (a,a) since those should be thrown out anyway")
    
  return basph_blocks2B  
       
def ph_block_constructor(subset2B, full_state, fermi_index, dim2B, dim1B):
    
  """fermi_index = particles[0]"""
  
  """ Mechanics of why this function identifies ph blocks:
  It goes over subset2B which has pairs (i,j) s.t i<j. It places (i,j) in blocks 
  according to its relative momenta, and spin signature : either even-even (0), odd-odd(0),
  odd-even (+1), even-odd(-1). It also computes the rel momenta and signature of (j,i).
  
  There are seemingly 4 possiblities:
    
    (1) blk in simple_ph_blocks & flip in simple_ph_blocks
    
    (2) blk not in simple_ph_blocks & flip in simple_ph_blocks
    
    (3) blk in simple_ph_blocks & flip not in simple_ph_blocks
    
    (4) blk not in simple_ph_blocks & flip not in simple_ph_blocks
    
  But possiblity (1) cannot occur. Why? In general,
  (1) can't be true because the algorithm works by first making blks in simple_ph_blocks 
  of the form [(i,j)] with i<j (in that case (1) can't be true since simple_ph_blocks[blks] 
  will not first exist ), then it appends (k,l) to those blks.  In that case, 
  since blks (i,j) and (j,i) are guaranted to be seperated (they have opposite momenta),
  by first picking blocks with i<j, I'll never find another blk with (j,i).
  This is slightly hard to explain. To better see this, look at the block in
  [(0, 6), (1, 7), (8, 0), (9, 1)] in basph_block2B for Nmax=0
  
  For ex, consider (0,13). It only connects
  with (2,1) for Nmax = 0. The loop will actually stumble upon (1,2) in subset2B
  before (0,13), so it will make a ph_block in simple_ph_blocks with (1,2) first. 
  Then, when it stumbles upon (0,13), flip can't be a block of simple_ph_blocks
  since 13 > 0 so it's instead added into the block made by (1,2). 
    
    """
  
  simple_ph_blocks = {}

  
  for subset_list in subset2B: #ph basis blks with (i,i)(j,j) will be zero relative
    #momentum blks, i.e only made up of particle-particle or hole-hole pairs, so ignore them
    
    for (i,j) in subset_list:
        
      sign = extra_symm_of_Ham(i,j) # group (i,j) so that it will only connect with similar spin pairs
      
      rel_mom = full_state[i][0:3] - full_state[j][0:3]
      
        
      blk = str(rel_mom) + str(sign) #relative momentum blk partitioned into smaller blks
      #with similar spins. See extra_symm_of_Ham(i,j) description.
      
      flip = str(full_state[j][0:3] - full_state[i][0:3]) + str(-sign)
           
            
      if blk not in simple_ph_blocks and flip not in simple_ph_blocks:
        
        simple_ph_blocks[blk] = [(i,j)]# form new block with given signature
        
        
      else:
        
        if flip in simple_ph_blocks:
          
          simple_ph_blocks[flip].append((j,i))
          
        else:
          
          simple_ph_blocks[blk].append((i,j))   
          
    
  return obtain_blocks(simple_ph_blocks, dim1B, dim2B, fermi_index)


def test_on_ph_refs(basph_blocks2B, full_state):
  
  Hermitian = list(range(len(basph_blocks2B)))
  
  for block_num1 in range(len(basph_blocks2B)):
    
    del Hermitian[Hermitian.index(block_num1)]
    
    for block_num2 in Hermitian:
                 
      for (i,j) in basph_blocks2B[block_num1]:
        
        for (k,l) in basph_blocks2B[block_num2]:
          
          if( np.all ( full_state[i][0:3]+full_state[l][0:3] == full_state[k][0:3]+full_state[j][0:3])):
            
            #print("Blocking algorithm for only momentum ph trans fails")
            
            if( i%2 + l%2 == k%2 + j%2):
            
              print("Blocking algorithm for mom + spins ph trans fails")
    
    
    
def special_index_ph(basph_block2B):#this is only for the ph transformation
  
  """The next function below does the same thing as idx2B, but stores the indices of pairs in a given block
  Figures out which block in which a given pair (p,q) lies. It also returns the index location of that 
  pair in bas_block2B: that is the row/column # of that given pair in Operator[block]"""
  
  #index = {} #dict containing pairs as keys to pair index -- not blocked
  
  index_blocked = {} #dict containing block_nums as keys to dicts wtih pairs as keys to pair index
  
  for block_num, bas in enumerate(basph_block2B):
  
    index_blocked[block_num] = {} 
    
    for pair in bas:
    
      ind = bas.index(pair)
      #index[pair] = (block_num, ind)
      index_blocked[block_num][pair] = ind
      
  return index_blocked
      
 # diagonal matrix: n_a - n_b
def construct_occupationA_2B(basph_block2B, occ1B):
  
  """ diagonal matrix: n_a - n_b 
  returns the occupation matrix only for the "reference pairs"
  "reference pairs" are the basis elements for the ph transformation that 
  I computed. There is a mirror set of pairs that I refer to as "mirrors". 
  These mirrors are obtained by flipping all "reference pairs". 
  So there really are 2*Num_ph_blocks. I choose not 
  to calculate the mirrors to save on storage. When I'm doing the matrix 
  products with ph_trans operators, I'll just slap a negative sign in front
  of the occupation matrix!
  
  Update: It turns out that those mirrored pairs are for computing 
  hermitian conjugate terms for resulting block operators. Since I know what
  those should be, I don't even need to explicity compute anything with the 
  mirrored terms. 
  """
  
  occ=[]#initialize list of block matrices for ph trans

  for blocks in basph_block2B:# go through blocks
    
    temp_occ_ref = np.zeros((len(blocks), len(blocks)))
      
    for i1, (i,j) in enumerate(blocks):
      
      temp_occ_ref[i1, i1] = occ1B[i] - occ1B[j]
      
    occ.append(temp_occ_ref)

  return occ

def ph_trans_then_mult(x, y, resultant, ph_block, block_occph, idp, sign):
  
  """ See description in ph_trans_prep() """
  
  #Perform PH Transform 
  
  x_ph = np.zeros((len(ph_block), len(ph_block)))  
  y_ph = np.zeros((len(ph_block), len(ph_block)))
  
  
  for i1, (a,b) in enumerate(ph_block):
    
    for i2, (c,d) in enumerate(ph_block):
      
      block_num, index_1 = idp[(a,d)]
      
      index_2 = idp[(c,b)][1]
      
      x_ph[ i1, i2 ] -= x[block_num][index_1, index_2]
      
      y_ph[ i1, i2 ] -= y[block_num][index_1, index_2]
      
      
  prod = dot(x_ph, dot(block_occph, y_ph))#do mult that you've waited all ur life to do 
  
  #Inverse Transform 
  
  for i1, (a,b) in enumerate(ph_block):
    
    for i2, (c,d) in enumerate(ph_block):
      
      block_num, index_1 = idp[(a,d)]; 
      
      #index_4 = idp[(d,a)][1]
            
      #index_3 = idp[(b,c)][1]; index_2 = idp[(c,b)][1]
      
      
      resultant[block_num][index_1, idp[(c,b)][1]] -= prod[ i1, i2 ]
      
      resultant[block_num][idp[(b,c)][1], idp[(d,a)][1]] = sign * -prod[ i1, i2 ] #conjugate term 
      
      # mirror contribute to hermitian conjugate of resultant_map
      
  return
      
def ph_trans_prep(x, y, basph_block2B, occphA_2B, bs_len, sign, idp, block_sizes):
  
  """ Takes in block operators x, y and preps them for conversion to ph representation. 
  Since both operators will have the same block structure in ph representation,
  their non-zero blocks are obtained using the blocks of basph_block2B.
  
  For each block of idxph2B, the function ph_trans_then_mult() obtains both x_ph[ph_block_num]
  and y_ph[ph_block_num], then multiplies them with the according occupation matrix.
  
  ph_trans_then_mult() then does the inverse transform to obtain resultant
  block matrix.
  
  """
  resultant = [np.zeros( (block_sizes[i], block_sizes[i]) ) for i in bs_len]#initialize list of block matrices
  
  #resultant_map = OP_Map("R")
  #resultant_map.obj = resultant
  
  for ph_block_num, ph_block in enumerate(basph_block2B):
    
    ph_trans_then_mult(x, y, resultant, ph_block, occphA_2B[ph_block_num], idp, sign)
    
  return resultant

def basic_checks(resultant, sign):
  
  if(sign == 1):

    for blocks in resultant:
      
      if(LA.norm( blocks - transpose(blocks)  ) != 0):
        
        print("Result is not hermitian")
        
  else:
    
    print(resultant[len(resultant)-23])
    
    #for blocks in resultant:
      #print(blocks)
      #if(LA.norm( blocks + transpose(blocks)  ) != 0):
        
        #print("Result is not anti-hermitian")
        #print( blocks + transpose(blocks)  )
    #print(blocks)
    #sys.exit()
        
def self_energy( Gamma,f, E, user_data ):
    
  #particles = user_data["particles"]
  #holes = user_data["holes"]
  states = user_data["states"]
  V = user_data["Gamma"]
  Particle_P = user_data["Particle_P"]
  Particle_A = user_data["Particle_A"]
  Hole_H = user_data["Hole_H"]
  Hole_A = user_data["Hole_A"]
  bs_len = user_data["bs_len"]
  
  Gamma_map = OP_Map_B("Gamma") #instance Gamma with attribute "Gamma"
  Gamma_map.obj = Gamma
  
  V_map = OP_Map_B("V") #instance intial Gamma with attribute "V"
  V_map.obj = V
  
  self_energy = np.zeros((len(states), len(states)))
  
  for p in states:
    for q in states:
      for block_num in bs_len:     
        for (i,r) in Hole_A[block_num]:
          for (j,s) in Hole_A[block_num]:
            if (i == j and r == p and s == q):
              for (a,b) in Particle_P[block_num]:
                  
                denom = E + f[i,i] - f[a,a] - f[b,b] 
                self_energy[p,q] = 0.5 * (Gamma_map[[(a,b), (i,p), block_num]] *
                           V_map[[(i,q), (a,b), block_num]])/denom    
 
        for (a,r) in Particle_A[block_num]:
          for (c,s) in Particle_A[block_num]:
            if (a == c and r == p and s == q):
              for (i,j) in Hole_H[block_num]:
                  
                denom = E + f[a,a] - f[i,i] - f[j,j] 
                self_energy[p,q] = 0.5 * (Gamma_map[[(i,j), (a,p), block_num]] *
                           V_map[[(a,q), (i,j), block_num]])/denom    
                             
                           
  return self_energy
#-----------------------------------------------------------------------------------
# commutator of matrices
#-----------------------------------------------------------------------------------
def commutator(a,b):
  return dot(a,b) - dot(b,a)

#-----------------------------------------------------------------------------------
# norms of off-diagonal Hamiltonian pieces and also, norm of 2-BD Lists
#-----------------------------------------------------------------------------------
def calc_fod_norm(f, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  
  norm = 0.0
  for a in particles:
    for i in holes:
      norm += f[a,i]**2 + f[i,a]**2

  return np.sqrt(norm)

def calc_Gammaod_norm(Gamma, user_data):
    
  #bs_len = user_data["bs_len"]
  #idp = user_data["idp"]
  idp_b    = user_data["idp_b"]
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  HH_PP_Filter = user_data["HH_PP_Filter"] 

  norm = 0.0
  
  for block_num in HH_PP_Filter:# go through blocks 
            
    for ket in Hole_H[block_num]:#no time will be wasted looping over Hole_H[blk#]
      
      i2 = idp_b[block_num][ket]
      
      for bra in Particle_P[block_num] :#if Particle_P[block_num] is empty,
          
        #i1 = idp_b[block_num][bra]
        norm += 8 * Gamma[block_num][idp_b[block_num][bra], i2]**2 

  return np.sqrt(norm)

#def calc_full2B_norm(Gamma, subset2B, subset_size, bs_len,idp_b):
#    
#  #Gamma_Map = OP_Map_B("Gamma")#create instance of OP_Map_B
#  #Gamma_Map.obj = Gamma #create class operator that it sees
#  
#  norm = 0.0
#  
#  for block_num in bs_len:# go through blocks
#      #find pairs in block      
#      
#      Hermitian = copy.copy(subset2B[block_num]) 
#      
#      for i1, bra in enumerate(subset2B[block_num]):
#        
#        del Hermitian[i1] #I'll manually do the diagonal terms
#        
#        for i2, ket in enumerate(subset2B[block_num]):
#          
#            norm += 4 * Gamma[block_num][ i1, i2]**2  
#
#  return np.sqrt(norm)

def calc_full2B_norm(Gamma, subset2B,bs_len,idp_b):#compare timing with other calc_full2B_norm()
    
  #Gamma_Map = OP_Map_B("Gamma")#create instance of OP_Map_B
  #Gamma_Map.obj = Gamma #create class operator that it sees
  
  norm = 0.0
  
  for block_num in bs_len:# go through blocks
      #find pairs in block      
      for i1, bra in enumerate(subset2B[block_num]):    
        for i2, ket in enumerate(subset2B[block_num]):
          
            norm += 4 * Gamma[block_num][ i1, i2]**2  

  return np.sqrt(norm)
#-----------------------------------------------------------------------------------
# Functions used to do algebraic manipulations on lists of matrices (2BD operators)
#-----------------------------------------------------------------------------------
def List_dot(A, B, block_sizes,bs_len): #performs matrix multiplication on blocks of A and B with same index
    return [dot(A[i],B[i]) for i in bs_len]

def List_operation(A, B, block_sizes,sign,operation, bs_len): #performs simple operations on lists
    
    if(operation == "Transpose"): #will transpose A
      return [transpose(A[i]) for i in bs_len]
  
    if(operation == "List_Add"): #will do addition and subtraction of lists based on sign given
      return [A[i] + sign * B[i] for i in bs_len] 
  
    if(operation == "Scalar_Mul"): #will do multiplication of list element (Matrix A[i]) with scalar (B)
      return [A[i] * B for i in bs_len]
  
#    if(operation == "Scalar_Div"): #will do division of list element (Matrix A[i]) with scalar (B)  
#      return [A[i]/B for i in bs_len]
#    if(operation != "Transpose" or operation != "List_Add" or operation != "Scalar_Mul"):
#        print("You fed in an undefined operation into List_Operation")
#        sys.exit()
 
#-----------------------------------------------------------------------------------
# occupation number matrices
#-----------------------------------------------------------------------------------
def construct_occupation_1B(dim1B, holes, particles):

  occ = np.zeros(dim1B)

  for i in holes:
    occ[i] = 1.

  return occ

# diagonal matrix: 1 - n_a - n_b
def construct_occupationB_2B(block_sizes, bas_block2B, occ1B, bs_len):
  occ=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

  for block_num in bs_len:# go through blocks
    for i1, (i,j) in enumerate(bas_block2B[block_num]):
      occ[block_num][i1, i1] = 1. - occ1B[i] - occ1B[j]

  return occ

# diagonal matrix: n_a * n_b 
def construct_occupationC_2B(block_sizes, bas_block2B, occ1B, bs_len):
  occ=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

  for block_num in bs_len:# go through blocks
    for i1, (i,j) in enumerate(bas_block2B[block_num]):
      occ[block_num][i1, i1] = occ1B[i] * occ1B[j]

  return occ 

#-----------------------------------------------------------------------------------
# generators
#-----------------------------------------------------------------------------------

def eta_white(f, Gamma, user_data):
  #dim1B     = user_data["dim1B"]
  #particles = user_data["particles"]
  #holes     = user_data["holes"]
  
  block_sizes = user_data["block_sizes"]
  bs_len = user_data["bs_len"]
  idp = user_data["idp"]
  idp_b    = user_data["idp_b"]
  subset_size = user_data["subset_sizes"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  
  HH_PP_Filter = user_data["HH_PP_Filter"] 
  
  # one-body part of the generator
#  eta1B  = np.zeros_like(f)
#  
#  for a in particles:
#    for i in holes:
#      denom = f[a,a] - f[i,i] + Gamma_Map_O[[(a,i),(a,i)]]
#      val = f[a,i]/denom
#      eta1B[a, i] =  val
#      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = [np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
  

  for block_num in HH_PP_Filter:# go through blocks
    
    disp = subset_size[block_num]# how far away to the right is (b,a) from (a,b)? same for (i,j)
    for (a,b) in Particle_P[block_num]:
      for (i,j) in Hole_H[block_num]:
        
        i1 = idp_b[block_num][(a,b)]#locaion of pair (a,b)
        i2 = idp_b[block_num][(i,j)]#location of pair (i,j)
        
        block_num1, index1 = idp[(a,i)]
        block_num2, index2 = idp[(a,j)]
        block_num3, index3 = idp[(b,i)]
        block_num4, index4 = idp[(b,j)]
        
#          if(block_num1 == block_num4):
#            print([(a,i),(b,j), block_num1])
        
        denom = ( 
        f[a,a] + f[b,b] - f[i,i] - f[j,j]  
        
        + Gamma[block_num][i1, i1]
        + Gamma[block_num][i2, i2]
        
        - Gamma[block_num1][index1 , index1] 
        - Gamma[block_num2][index2 , index2] 
        - Gamma[block_num3][index3 , index3] 
        - Gamma[block_num4][index4 , index4] 
        )
        val = Gamma[block_num][i1, i2] / denom
        #val = Gamma_Map[[(a,b),(i,j),block_num]] / denom

        eta2B[block_num][i1, i2] = val
        #eta2B_map[[(a,b),(i,j),block_num]] = val
        eta2B[block_num][i2, i1] = -val
        #eta2B_map[[(i,j),(a,b),block_num]] = -val
        
        eta2B[block_num][i1 + disp, i2] = -val
        #eta2B_map[[(b,a),(i,j),block_num]] = -val
        eta2B[block_num][i2, i1 + disp] = val
        #eta2B_map[[(i,j),(b,a),block_num]] = val
        
        eta2B[block_num][i1, i2 + disp] = -val
        #eta2B_map[[(a,b),(j,i),block_num]] = -val
        eta2B[block_num][i2 + disp, i1] = val
        #eta2B_map[[(j,i),(a,b),block_num]] = val
        
        eta2B[block_num][i1 + disp, i2 + disp] = val
        #eta2B_map[[(b,a),(j,i),block_num]] = val
        eta2B[block_num][i2 + disp, i1 + disp] = -val
        #eta2B_map[[(j,i),(b,a),block_num]] = -val

  return eta2B

def Bernoulli_generator(y): #returns list of Bernoulli numbers indexed by 0 to (y-1)
    def bernoulli2(): # Function taken online to calculate Bernoulli sequence
        A, m = [], 0
        while True:
            A.append(Fr(1, m+1))
            for j in range(m, 0, -1):
              A[j-1] = j*(A[j-1] - A[j])
            yield A[0] # (which is Bm)
            m += 1
    bn2 = [ix for ix in zip(range(y), bernoulli2())]
    bn3=[]
    for u in range(len(bn2)): # Convert messy format into easily accessible array 
        bn3.append(float(bn2[u][1]))
    return bn3

#insert remaning generators when you're ready to adapt them. 

#-----------------------------------------------------------------------------------
# Derivative routines
#-----------------------------------------------------------------------------------
#@jit  
def special_commutator(x, y, user_data,sign): # takes in either 1 or 2 body matrices to output 0,1, and 2 body commutators
#sign ensures that proper hermiticity is preserved in the particle hole transformation
#Since this commuatator is used for both the Hamiltonian calcuation with terms that look like ~[H,Omega] (hermitian with anti-hermitian)
#and Magnus expansion with terms that look like ~[Omega, eta] (anti-hermitian with anti-hermitian), the definition of xy in the 2B-2B
#commutator must be such that ~[H,Omega] returns a hermitian matrix; and ~[Omega, eta] returns an anti-hermitian matrix
                   
    if(type(x) == list and type(y) == list):# 2B-2B
          
      
        idp       = user_data["idp"]
        #bas2B     = user_data["bas2B"]
        occB_2B   = user_data["occB_2B"]
        occC_2B   = user_data["occC_2B"]
        occphA_2B = user_data["occphA_2B"]
        
        subset_size = user_data["subset_sizes"]
        block_sizes = user_data["block_sizes"]
        bs_len = user_data["bs_len"]
        
        subset2B = user_data["subset2B"]
        basph_block2B = user_data["basph_block2B"]  
        
        
        xy= List_dot(x, List_dot(occB_2B, y, block_sizes, bs_len), block_sizes, bs_len)# xy has same blk structure
        xyz= List_dot(x, List_dot(occC_2B, y, block_sizes, bs_len), block_sizes, bs_len)#xyz has same blk structure
        
        #Output_2B= 0.5 * (xy + sign * transpose(xy))#sign reflects proper hermiticity
        
        Transpose = List_operation(xy, "NA", block_sizes,sign,"Transpose", bs_len)
        bracket = List_operation(xy, Transpose, block_sizes,sign,"List_Add", bs_len)        
        
#        x_map_o = OP_Map("x") #instance x with attribute "x"
#        y_map_o = OP_Map("y") #instance y with attribute "y"
#        x_map_o.obj = x #set appropriate x object 
#        y_map_o.obj = y #set appropriate y object

        xyi = ph_trans_prep(x, y, basph_block2B, occphA_2B, bs_len, sign, idp, block_sizes)
        Output_2B = List_operation(bracket, 0.5, block_sizes,sign,"Scalar_Mul", bs_len)
        
        if(sign == 1): #if commutator coming from Hamiltonian routine
          
            dim1B     = user_data["dim1B"]
            idp_b    = user_data["idp_b"]
            
            #Initializations 
            Output_0B = 0.0
            Output_1B = np.zeros((dim1B, dim1B))
            
            Particle_P = user_data["Particle_P"]
            Hole_H = user_data["Hole_H"]           
            Hole_A = user_data["Hole_A"]
            
            HH_PP_Filter = user_data["HH_PP_Filter"] 
            Hole_A_Filter = user_data["Hole_A_Filter"]
                                          
            
            for block_num in HH_PP_Filter:# go through blocks
              
                for (i,j) in Hole_H[block_num]:
                    
                    i1 =  idp_b[block_num][(i,j)]#I can reuse this
                                      
                    for (k,l) in Particle_P[block_num]:  
                      
                        i2 =  idp_b[block_num][(k,l)]
                                                
                        Output_0B += 2*(
                            x[block_num][i1, i2]
                            *y[block_num][i2, i1])
                        
                                       
            for block_num in Hole_A_Filter:# go through blocks
            
                for (i,p) in Hole_A[block_num]:
                    
                    i1 =  idp_b[block_num][(i,p)]
                    
                    Output_1B[p, p] += xy[block_num][i1, i1] 
                
#                for (r,p) in subset2B[block_num]:
#                  
#                    Output_1B[p, p] +=  xyz_map[[(r,p), (r,p), block_num]]
#                    
#                    Output_1B[r, r] += xyz_map[[(p,r), (p,r),block_num]] 
                    
                    
            for block_num in bs_len:# go through blocks    
                
                disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)      
                
                Hermitian = copy.copy(subset2B[block_num]) 
                
                for i1, (r,p) in reverse_enumerate(Hermitian,disp):
                  
                    index = idp_b[block_num][(r,p)]#locaion of pair (r,p)
                  
                    Output_1B[p, p] +=  xyz[block_num][index, index]
                    
                    Output_1B[r, r] +=  xyz[block_num][index + disp, index + disp]
                    
                    for i2, (j,q) in enumerate(Hermitian):
    
                            
                        Output_2B[block_num][i1, i2]-=( 
                                xyi[block_num][i1, i2]
                                - xyi[block_num][i1 + disp, i2]
                                - xyi[block_num][i1, i2 + disp]
                                + xyi[block_num][i1 + disp, i2 + disp])
                        
                        mval = Output_2B[block_num][i1, i2] #I hope this gets put in cache
                        Output_2B[block_num][i1 + disp, i2]= -mval
                        Output_2B[block_num][i1, i2 + disp]= -mval
                        Output_2B[block_num][i1 + disp, i2 + disp]= mval
                        
                        Output_2B[block_num][i2, i1] =  mval
                        Output_2B[block_num][i2, i1 + disp] = -mval
                        Output_2B[block_num][i2 + disp, i1] = -mval
                        Output_2B[block_num][i2 + disp, i1 + disp] = mval
                        
                    del Hermitian[i1] 
                    #delete (r,p) so it does show up in Hermitian since transpose is manually coded                        

            return Output_0B, Output_1B, Output_2B
          
          
        else: #Omega_0B and Omega_1B are zeros, so only worry about 2BD piece in RHS routine
          
            for block_num in bs_len:# go through blocks
                
                disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)                 
                Hermitian = copy.copy(subset2B[block_num]) 
                
                for i1, (r,p) in reverse_enumerate(Hermitian,disp):
                    for i2, (j,q) in enumerate(Hermitian):
    
                        Output_2B[block_num][i1, i2]-=( 
                                xyi[block_num][i1, i2]
                                - xyi[block_num][i1 + disp, i2]
                                - xyi[block_num][i1, i2 + disp]
                                + xyi[block_num][i1 + disp, i2 + disp])
                        
                        mval = Output_2B[block_num][i1, i2] #I hope this gets put in cache
                        Output_2B[block_num][i1 + disp, i2]= -mval
                        Output_2B[block_num][i1, i2 + disp]= -mval
                        Output_2B[block_num][i1 + disp, i2 + disp]= mval
                        
                        Output_2B[block_num][i2, i1]= -mval
                        Output_2B[block_num][i2, i1 + disp]= mval
                        Output_2B[block_num][i2 + disp, i1]= mval
                        Output_2B[block_num][i2 + disp, i1 + disp]= -mval
                        
                    del Hermitian[i1] 
                    #delete (r,p) so it does show up in Hermitian since transpose is manually coded    
                    
            return Output_2B   
      
      
    
                    
    if(type(x) == list and type(y) == np.ndarray):#2B-1B 
        
          
        subset_size = user_data["subset_sizes"]
        block_sizes = user_data["block_sizes"]
        bs_len = user_data["bs_len"]
        subset2B = user_data["subset2B"]
        
        
        Output_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
        
                    
        for block_num in bs_len:# go through blocks  
          
            disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)    
            Hermitian = copy.copy(subset2B[block_num])  
            
            for i1, (i,j) in reverse_enumerate(Hermitian,disp):
                for i2, (k,l) in enumerate(Hermitian):                  
                    
                    
                    mval = -y[l,l] * x[block_num][i1, i2 + disp] #I hope this is stored in cache
                    
                    mval += (y[k,k] - y[i,i]) * x[block_num][i1, i2]
                    
                    mval += y[j,j] * x[block_num][i1 + disp, i2]
                    
                    #mval -= y[i,i] * x[block_num][i1, i2]
                                      
                        
                    Output_2B[block_num][i1, i2] = mval 
                    Output_2B[block_num][i1 + disp, i2]= -mval
                    Output_2B[block_num][i1, i2 + disp]= -mval
                    Output_2B[block_num][i1 + disp, i2 + disp]= mval
                    
                    Output_2B[block_num][i2, i1]= mval
                    Output_2B[block_num][i2, i1 + disp]= -mval
                    Output_2B[block_num][i2 + disp, i1]= -mval
                    Output_2B[block_num][i2 + disp, i1 + disp]= mval
                    
                del Hermitian[i1] 
                #delete (i,j) so it does show up in Hermitian since transpose is manually coded
                    
        return Output_2B       

def RHS_Cal(Omega_2B,Eta_2B, user_data):# Generates right hand side of flow equation to be fed to diffy solver to obtain new Omega
    
    bn2 = user_data["bn2"]
    block_sizes = user_data["block_sizes"]
    subset2B = user_data["subset2B"]
    bs_len = user_data["bs_len"]
    #idp = user_data["idp"]
    idp_b    = user_data["idp_b"]
    
    #RHS_0B=0. #Initialize RHS
    #RHS_1B=np.zeros_like(Omega_1B)
    RHS_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

    for n in range(len(bn2)):
        if n==0: 
            nth_2B = Eta_2B #Initial commutators defined to be Eta
            #nth_1B=copy.copy(Eta_1B)#eta1B=0

        else:
            #nth_1B will always be zero, see latex document [\eta^{2BD}(0),\,\,\eta^{2BD}(ds)]^{1BD}=0
            #C_1B_1B=special_commutator(Omega_1B,nth_1B, user_data, -1)#store one body-one body commutator
            #C_1B_2B=special_commutator(Omega_1B,nth_2B, user_data, -1)#store one body-two body commutator
            #C_2B_1B=special_commutator(Omega_2B,nth_1B, user_data, -1)#store two body-one body commutator
            nth_2B=special_commutator(Omega_2B,nth_2B, user_data, -1)#store two body-two body commutator
            #; -1 for anti-hermitian 2B piece
            # The next lines aggregate the current 0B, 1B, and 2B nth commutators from the Magnus expansion
      
            #nth_1B=C_1B_1B[1]+C_1B_2B[1]+C_2B_1B[1]+C_2B_2B[1] #extract one body terms
            
            #First_add=List_operation(C_2B_1B[2], C_2B_2B[2], block_sizes,+1,"List_Add", bs_len)
            #nth_2B=List_operation(C_1B_2B[2], First_add, block_sizes,+1,"List_Add", bs_len)#extract two body terms
            
            #nth_2B = C_2B_2B
            
        
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        #shiftymatrix_0B=nth_0B/np.math.factorial(n)
        #shiftymatrix_1B=nth_1B/np.math.factorial(n)
        
        shiftymatrix_2B=List_operation(nth_2B, 1/np.math.factorial(n), block_sizes,"NA","Scalar_Mul", bs_len)
        
        if (calc_full2B_norm(shiftymatrix_2B, subset2B,bs_len,idp_b) < 1e-8 ):
            break        
        #RHS_1B+=bn2[n]*shiftymatrix_1B
        
        First_mul = List_operation(shiftymatrix_2B, bn2[n], block_sizes,"NA","Scalar_Mul", bs_len)
        RHS_2B = List_operation(RHS_2B, First_mul, block_sizes,+1,"List_Add", bs_len)

    return  RHS_2B

def Transformed_Ham(Omega_2B, user_data):# Generates new Hamiltonian by performing BCH expansion
  
    bn2 = user_data["bn2"]
    block_sizes = user_data["block_sizes"]
    #subset2B = user_data["subset2B"]
    E=user_data["E"]
    f=user_data["f"]
    Gamma=user_data["Gamma"]
    bs_len = user_data["bs_len"]
    #idp = user_data["idp"]
    #idp_b    = user_data["idp_b"]
    
    H_0B=0. #Initialize Hamilitonian
    H_1B=np.zeros_like(f)
    H_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

    for n in range(len(bn2)):
        if n==0: 
            nth_2B = copy.deepcopy(Gamma) #Initial commutators defined to be H
            nth_1B = copy.copy(f)
            nth_0B = E 
        else:
            #C_1B_1B=special_commutator(Omega_1B,nth_1B, user_data, +1)#store one body-one body commutator
            #C_1B_2B=special_commutator(Omega_1B,nth_2B, user_data, +1)#store one body-two body commutator
            C_2B_1B=special_commutator(Omega_2B,nth_1B, user_data, +1)#store two body-one body commutator
            C_2B_2B=special_commutator(Omega_2B,nth_2B, user_data, +1)#store two body-two body commutator
            #; +1 for hermitian 2B piece
            # The next lines aggregate the current 0B, 1B, and 2B nth commutators from the Magnus expansion
            nth_0B = C_2B_2B[0] #extract zero body terms       
            nth_1B = C_2B_2B[1] #extract one body terms
            
            nth_2B=List_operation(C_2B_1B, C_2B_2B[2], block_sizes,+1,"List_Add", bs_len)
            #nth_2B=List_operation(C_1B_2B[2], First_add, block_sizes,+1,"List_Add", bs_len)#extract two body terms
            
            
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        shiftymatrix_0B=nth_0B/np.math.factorial(n)
        shiftymatrix_1B=nth_1B/np.math.factorial(n)
        
        shiftymatrix_2B=List_operation(nth_2B, 1/np.math.factorial(n), block_sizes,"NA","Scalar_Mul", bs_len)
        #if ((LA.norm(shiftymatrix_1B)+ calc_full2B_norm(shiftymatrix_2B, subset2B,bs_len,idp_b)) < 1e-10 ):
        if (abs(shiftymatrix_0B) < 1e-8 ):#MeV
            #print(n)
            break        
        H_0B+=shiftymatrix_0B
        H_1B+=shiftymatrix_1B
        
        H_2B = List_operation(H_2B, shiftymatrix_2B, block_sizes,+1,"List_Add", bs_len)
        
    return H_0B, H_1B, H_2B
#------------------------------------------------------------------------------
# derivative wrapper
#------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, block_sizes, bs_len):
  
  #block_sizes = user_data["block_sizes"]
  two_body=[]
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
#  zero_body = y[ptr]
#
#  ptr += 1
#  one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))
#  ptr += dim1B*dim1B
  
  for i in bs_len:
      #dimBlk=block_sizes[i]*block_sizes[i]
      two_body.append(reshape(y[ptr : ptr+block_sizes[i]*block_sizes[i]], (block_sizes[i], block_sizes[i])))
      ptr += block_sizes[i]*block_sizes[i]

  return two_body

def List_reshape(dOmega_2B,block_sizes, bs_len): #flatten 2B list into one array
    Output=[]
    for i in bs_len:
        Output.extend(reshape(dOmega_2B[i], -1))
        
    return np.array(Output)

def derivative_wrapper(y, user_data, Transformed_Hamiltonian):

  dim1B = user_data["dim1B"]
  #dim2B = dim1B*dim1B
  calc_eta  = user_data["calc_eta"]# Function to calculate generator
  calc_rhs  = user_data["calc_rhs"]    #function to calculate RHS
  subset2B = user_data["subset2B"]
  block_sizes = user_data["block_sizes"]
  bs_len = user_data["bs_len"]
  #idp = user_data["idp"]
  idp_b    = user_data["idp_b"]
  
  # extract operator pieces from solution vector
  Omega_2B = get_operator_from_y(y, dim1B, block_sizes, bs_len)

  # calculate the 2BD generator
  Eta_2B = calc_eta(Transformed_Hamiltonian[1], Transformed_Hamiltonian[2], user_data)
  
  # share data
  #user_data["dE"] = dOmega_0B #storing dOmega_0B/ds in dE
  user_data["eta_norm"] =  calc_full2B_norm(Eta_2B, subset2B,bs_len,idp_b)  
  
  # calculate the right-hand side
  dOmega_2B = calc_rhs(Omega_2B, Eta_2B, user_data)

  # convert derivatives into linear array
  dy   = List_reshape(dOmega_2B,block_sizes, bs_len)
  
  return dy

#-----------------------------------------------------------------------------------
# Infinite Matter Hamiltonian
#-----------------------------------------------------------------------------------
def radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,alpha,L):#returns radial matrix elements (with no definite symm)
    #alpha corresponds to index of the arrays k_alpha and Combined_consts
    #p,q,r,s correspond to single particle states with wavevectors
    q = np.pi/L * ( full_state[p][0:3] - full_state[q][0:3] + full_state[s][0:3] - full_state[r][0:3])
    return Combined_consts[alpha] * np.exp(-np.dot(q,q)/(4 * k_alpha[alpha] ))

def spin_exch_element(p,q,r,s,full_state,ind):#returns spin matrix elements (with no definite symm)

    #ind=3 means that regular spin elements are computed; ind=4 means isospin elements are computed
    Element=0# intialize spin matrix element
    if(full_state[p][ind]==full_state[r][ind] and full_state[q][ind]==full_state[s][ind] ):
        Element+=0.5 + 2 * full_state[s][ind] * full_state[r][ind]
        
    if(full_state[p][ind]==full_state[r][ind]-1 and full_state[q][ind]==full_state[s][ind]+1 ):
        Element+=np.sqrt(3/4-full_state[r][ind] * (full_state[r][ind]-1)) * np.sqrt(3/4-full_state[s][ind] * (full_state[s][ind]+1))
    
    if(full_state[p][ind]==full_state[r][ind]+1 and full_state[q][ind]==full_state[s][ind]-1 ):
        Element+=np.sqrt(3/4-full_state[r][ind] * (full_state[r][ind]+1)) * np.sqrt(3/4-full_state[s][ind] * (full_state[s][ind]-1))
        
    return Element
    
def H2B_element(p,q,r,s,full_state,Combined_consts, k_alpha,degen,L):
    
    #if degen=1,then <t_pt_q|P^t|t_rt_s>=1 ->actually an isospin triplet state
    Element=0; A=0#These form the main pieces of the matrix element H2B[idx2B[(p,q)],idx2B[(r,s)]]
    spin = spin_exch_element(p,q,r,s,full_state,3)#precompute spin matrix elements
    if(degen == 4):#if I'm including isospin
        isospin= spin_exch_element(p,q,r,s,full_state,4)#precompute isospin matrix elements
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3] 
        and full_state[p][4]==full_state[r][4] and full_state[q][4]==full_state[s][4]):
            A+=1
        A-= spin * isospin
        Element+=0.5 * A * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,0,L)#R term
        
        B=A; C=A
        if(full_state[p][4]==full_state[r][4] and full_state[q][4]==full_state[s][4]):
            B+= spin; C-= spin
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3]):
            B-= isospin; C+= isospin
        Element+=0.25 * B * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,1,L)#T term
        Element+=0.25 * C * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,2,L)#S term
        
    if(degen == 2):# if I'm not including isospin
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3]):
            A+=1
        A-= spin
        Element+=0.5 * A * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,0,L)#R term
        B = A + spin; C = A - spin
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3]):
            B-= 1; C+= 1
        Element+=0.25 * B * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,1,L)#T term
        Element+=0.25 * C * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,2,L)#S term
        
    
    return Element #matrix element with no definite symmetry
    

    
def Inf_Matter_Ham(full_state, energy_const, Combined_consts, k_alpha, degen, L, user_data):
    
    states = user_data["states"]
    #bas2B = user_data["bas2B"]
    subset2B = user_data["subset2B"]
    block_sizes = user_data["block_sizes"]
    bas_block2B = user_data["bas_block2B"]
    bs_len = user_data["bs_len"]

    H1B=np.zeros((len(full_state), len(full_state)))
    for i in states:
        H1B[i,i] = energy_const * np.dot(full_state[i][0:3], full_state[i][0:3]) #0:3 doesn't include 3
        
    H2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
    for blocky in bs_len:# fill in all blocks
        #print(block)
        Hermitian=copy.copy(subset2B[blocky])# I plan to exploit H2B's hermiticity: 
        #Once a pair (p,q) has been considered in the bra, then it won't be considered 
        #in the ket via my loop. I will manually compute the transpose.
        for (p,q) in subset2B[blocky]:
            for (r,s) in Hermitian:
                #print([(p,q),(r,s)])
                
                #Get respective indices of pairs in each block
                block_loc_pq=bas_block2B[blocky].index((p,q))
                block_loc_rs=bas_block2B[blocky].index((r,s))
                block_loc_qp=bas_block2B[blocky].index((q,p))
                block_loc_sr=bas_block2B[blocky].index((s,r))
                
                #Compute anti-symmetrized matrix element <(p,q)| H2B |(r,s)>_AS
                if (np.all( full_state[p][0:3] + full_state[q][0:3] != full_state[r][0:3] + full_state[s][0:3] )):
                    print("CM Momentum is not being conserved!")
                #ensure that center of mass momentum is conserved for all matrix elements
                H2B[blocky][block_loc_pq,block_loc_rs] = (H2B_element(p,q,r,s,full_state,Combined_consts, k_alpha,degen,L)
                -H2B_element(p,q,s,r,full_state,Combined_consts, k_alpha,degen,L))#obtain anti-symm matrix element
                #Use properties of anti-symmetrized matrix elements
                H2B[blocky][block_loc_pq,block_loc_sr]= -H2B[blocky][block_loc_pq,block_loc_rs]
                H2B[blocky][block_loc_qp,block_loc_rs]= -H2B[blocky][block_loc_pq,block_loc_rs]
                H2B[blocky][block_loc_qp,block_loc_sr]=  H2B[blocky][block_loc_pq,block_loc_rs]
                
                #Use Hermiticity, also note that matrix elements are real
                H2B[blocky][block_loc_rs,block_loc_pq]= H2B[blocky][block_loc_pq,block_loc_rs]
                H2B[blocky][block_loc_sr,block_loc_pq]= H2B[blocky][block_loc_pq,block_loc_sr]
                H2B[blocky][block_loc_rs,block_loc_qp]= H2B[blocky][block_loc_qp,block_loc_rs]
                H2B[blocky][block_loc_sr,block_loc_qp]= H2B[blocky][block_loc_qp,block_loc_sr]
                
            del Hermitian[Hermitian.index((p,q))] #delete (p,q) so it does show up in Hermitian since transpose is manually coded
                    
    return H1B, H2B
 
#-----------------------------------------------------------------------------------
# Normal-ordered Infinite Matter Hamiltonian
#-----------------------------------------------------------------------------------
def normal_order(H1B, H2B, user_data):
  #This will rewrite the already second quantized Inf_matter Hamiltonian=H1B+H2B in terms of the Hartree Fock energy, the Fock operator, and H2B
  holes = user_data["holes"]
  
  Hole_H = user_data["Hole_H"]#used to get 2B contribution to E in normal order routine
  Hole_A = user_data["Hole_A"]#used to get 2B contribution to E in normal order routine
  
  idp_b    = user_data["idp_b"]
  bs_len = user_data["bs_len"]
  
  # 0B part
  E = 0.0
  
  # 1B part 
  f = copy.copy(H1B)
  
  for i in holes:
    
    E += H1B[i,i]
    
  for block_num in bs_len:
 #subset_holes has pairs (i,j) for i<j  --> allowed by exclusion principle.
# I add the corresponding (j,i) term using symmetry:
 #Since H2B[idx2B[(i,j)],idx2B[(i,j)]]=H2B[idx2B[(j,i)],idx2B[(j,i)]], 
 #I multiply  0.5*H2B[idx2B[p],idx2B[p]]
 #by a factor of 2. This way, I have a smaller loop range by a factor of 2.     
   for (a,b) in Hole_H[block_num]:
      
     i1 = idp_b[block_num][(a,b)]
     
     E += H2B[block_num][i1, i1]

   for (i,p) in Hole_A[block_num]: #mean field contributions
    
     i1 = idp_b[block_num][(i,p)]
     
     f[p, p] += H2B[block_num][i1, i1]  #since (i,p) and (i,q) are in the same 
     #Blk, p = q

  # 2B piece of new normal ordered Hamiltonian is H2B. No need for corrections to H2B

  return E, f, H2B

#def Cheap_Self_Energy(E, Gamma, f, particles, dim1B):
#  Sigma = np.zeros (dim1B, dim1B)
#-----------------------------------------------------------------------------------
# Perturbation theory
#-----------------------------------------------------------------------------------
def calc_mbpt2(f, Gamma, user_data):
  
  DE2 = 0.0
  bs_len = user_data["bs_len"]
  #idp    = user_data["idp"]
  idp_b    = user_data["idp_b"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  
  for block_num in bs_len:# go through blocks
      
    if Particle_P[block_num]:
      
      for (i,j) in Hole_H[block_num]:
      
        index = idp_b[block_num][(i,j)]# I can reuse this
        
        for (a,b) in Particle_P[block_num]:
        
          
          #i1 = idp_b[block_num][(a,b)]
          #i2 = idp_b[block_num][(i,j)]
          
          denom = f[i,i] + f[j,j] - f[a,a] - f[b,b]
          me    = Gamma[block_num][idp_b[block_num][(a,b)], index]
          DE2  += me*me/denom

  return DE2

def calc_mbpt3(f, Gamma, user_data):

  particles = user_data["particles"]
  holes     = user_data["holes"]
  bs_len = user_data["bs_len"]
  #idp    = user_data["idp"]
  #idp_b    = user_data["idp_b"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  
  Gamma_Map = OP_Map_B("Gamma") #instance of OP_Map with attribute "Gamma"
  Gamma_Map.obj = Gamma
  
  Gamma_Map_O = OP_Map("Gamma")#create instance of OP_Map
  Gamma_Map_O.obj = Gamma #create class operator that it sees
  
  DE3pp = 0.0
  DE3hh = 0.0
  DE3ph = 0.0
  
  for block_num in bs_len:# go through blocks
    
    for (i,j) in Hole_H[block_num]:    
      for (a,b) in Particle_P[block_num]:
        
        for (c,d) in Particle_P[block_num]:
          denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[i,i] + f[j,j] - f[c,c] - f[d,d])
          me    = (Gamma_Map[[(i,j),(a,b),block_num]]
          *Gamma_Map[[(a,b),(c,d),block_num]]
          *Gamma_Map[[(c,d),(i,j),block_num]])
          DE3pp += me/denom    
                         
        for (k,l) in Hole_H[block_num]:
          denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[l,l] - f[a,a] - f[b,b])
          me    = (Gamma_Map[[(a,b),(k,l),block_num]]
          *Gamma_Map[[(k,l),(i,j),block_num]]
          *Gamma_Map[[(i,j),(a,b),block_num]])
          DE3hh += me/denom  
          
        for k in holes:
          for c in particles:
            denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[j,j] - f[a,a] - f[c,c])
            me    = (Gamma_Map[[(i,j),(a,b),block_num]]#original expression
            *Gamma_Map_O[[(k,b),(i,c)]]
            *Gamma_Map_O[[(a,c),(k,j)]])
            DE3ph -= me/denom

            denom = (f[i,i] + f[j,j] - f[b,b] - f[a,a])*(f[k,k] + f[j,j] - f[b,b] - f[c,c]) 
            me    = (Gamma_Map[[(i,j),(b,a),block_num]]#flip (a,b)
            *Gamma_Map_O[[(k,a),(i,c)]]
            *Gamma_Map_O[[(b,c),(k,j)]])
            DE3ph -= me/denom

            denom = (f[j,j] + f[i,i] - f[a,a] - f[b,b])*(f[k,k] + f[i,i] - f[a,a] - f[c,c])
            me    = (Gamma_Map[[(j,i),(a,b),block_num]]#flip (i,j)
            *Gamma_Map_O[[(k,b),(j,c)]]
            *Gamma_Map_O[[(a,c),(k,i)]])
            DE3ph -= me/denom

            denom = (f[j,j] + f[i,i] - f[b,b] - f[a,a])*(f[k,k] + f[i,i] - f[b,b] - f[c,c])
            me    = (Gamma_Map[[(j,i),(b,a),block_num]]#flip (a,b) and (i,j)
            *Gamma_Map_O[[(k,a),(j,c)]]
            *Gamma_Map_O[[(b,c),(k,i)]])
            DE3ph -= me/denom
            
  return DE3pp+DE3hh+DE3ph

def time_test_for_map(Gamma, idp_b, subset2B, full_state, mom_to_blk_nums, idp): 
  
  "seems like OP_Map_B is twice as expensive as just index directly using idp_b"
  gam = OP_Map_B("gam")
  gam.obj = Gamma
  yys = 0
  run = 100
  lenny = len(subset2B)
  start_time = time.time()
  
  for it in range(run):
    for block_num in range(lenny):
      
      for (a,b) in subset2B[block_num]:
        for (c,d) in subset2B[block_num]:
          
          yys = gam[[(a,b), (c,d), block_num]]
          
  print("Using OP_Map_B",time.time() - start_time)
 
  
  start_time = time.time()
  
  for it in range(run):
    for block_num in range(lenny):
      
      for (a,b) in subset2B[block_num]:
        for (c,d) in subset2B[block_num]:
          
          i1 = idp_b[block_num][(a,b)]#locaion of pair (a,b)
          i2 = idp_b[block_num][(c,d)]#location of pair (i,j)
          yys = Gamma[block_num][i1, i2]
          
  print("Using idp_B",time.time() - start_time)
  
  
#  start_time = time.time()
#  
#  for it in range(run):
#    for block_num in range(lenny):
#      
#      for (a,b) in subset2B[block_num]:
#        
#        mom = full_state[a][0:3] + full_state[b][0:3]
#        
#        spin = a%2 + b%2
#        block_num1 = mom_to_blk_nums[ str(mom) + str(spin) ]
#        
#        for (c,d) in subset2B[block_num]:
#          
#        
#                                        
#          i1 = idp_b[block_num1][(a,b)]#locaion of pair (a,b)
#          i2 = idp_b[block_num1][(c,d)]#location of pair (i,j)
#          
#          
#          yys = Gamma[block_num1][i1, i2]
#          
#  print("Using mom_to_blks_nums",time.time() - start_time)
  

  start_time = time.time()
  
  for it in range(run):
    for block_num in range(lenny):
      
      for (a,b) in subset2B[block_num]:        
        
        
         
        for (c,d) in subset2B[block_num]:
          
          block_num1, i1 = idp[(a,b)]
          i2 = idp[(c,d)][1]#location of pair (i,j)
          
          
          yys = Gamma[block_num1][i1, i2]
          
  print("Using just idp",time.time() - start_time)
  
def Omega_checker(Omega, Hole_H, Particle_P, bs_len):
   
   
  for block_num in bs_len:
     
    if(not Hole_H[block_num] or not Particle_P[block_num]):
       
      if(LA.norm(Omega[block_num]) != 0.0):
         
        print(LA.norm(Omega[block_num]))
        
#    if( Hole_H[block_num] and Particle_P[block_num]):
#       
#         
#      print(LA.norm(Omega[block_num]))  
def worm(Omega, bs_len, s, countering):

  for block_num in bs_len:

    if(LA.norm(Omega[block_num]) == 0.0 and s!=0):
         
      countering.append(block_num)
       
def fermi_understanding(bas_block2B, fermi_index):

  for blocks in bas_block2B:

    first = blocks[0]        
    
    type_blk = []
    
    if(first[0]< fermi_index and first[1]< fermi_index):
      
      type_blk.append("hh")
      
    if(first[0]< fermi_index and first[1]>= fermi_index):
      
      type_blk.append("hp")

    if(first[0]>= fermi_index and first[1]< fermi_index):
      
      type_blk.append("ph")

    if(first[0]>= fermi_index and first[1]>= fermi_index):
      
      type_blk.append("pp")
      

    for (a,b) in blocks:
      
      if(a< fermi_index and b <fermi_index):
        
        type_blk.append("hh")
        
      if(a< fermi_index and b>= fermi_index):
        
        type_blk.append("hp")
  
      if(a>= fermi_index and b< fermi_index):
        
        type_blk.append("ph")
  
      if(a>= fermi_index and b>= fermi_index):
        
        type_blk.append("pp")
        
        
#    if("ph" in type_blk and "pp" in type_blk):
#      
#      print("ph and pp are in same blk")
#      print(blocks)
#      sys.exit()
      
    if("ph" in type_blk and "hh" in type_blk):
      
      print("ph and hh are in same blk")
      print(blocks)
      sys.exit()  
      
      
#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

def main():

  ####Can change N_Max (determines # of particles),rho (determines size of box and Fermi lvl), and degen (type of matter) ######
  N_Max = 5 #Define N_Max
  rho = 0.2 #Define density in fm^-3
  degen = 2#If degen=4, then the matter is symmetric and isospin projections are considered. If degen=2, there is only
  #one type of particle, so isospin projections are not considered. Degen is only either 2 or 4. 

  temp_full_state = full_state_gen_3D(N_Max+1,degen)#this contains the "positive states" of the system
  #Notice that N_Max is shifted up by one to generate particle states. The true N_Max is still the one given above.
  #As the way it is coded now, all particle states above the given N_Max are used. Perhaps, it in the future, it may be best to 
  #Select the states above the given N_Max that I want to use for the calcuation. Just a thought....

  #############################

  full_state = sorter(temp_full_state, N_Max)#order the states so that holes come before particle states in the list. I like it that way
  full_state=np.array(full_state)#covert it to an array for easy viewing
  #Note that full state corresponds to single particle states labled by mode #'s n, not wavenumbers k

  fully.append(full_state)
  holes, particles=One_BD_States(N_Max, full_state)
  
  states = holes + particles
  A=len(holes) 
  L=(A/rho)**(1/3)
  print("Box Size (fm): ", L)
  print("Number of particles: ", A)
  
  Combined_consts=[(200/L**3)*(np.pi/1.487)**(3/2), -(178/L**3)*(np.pi/0.639)**(3/2), -(91.85/L**3)*(np.pi/0.465)**(3/2)]
  #Combined constants for matrix element. 
  k_alpha=[1.487,0.639,0.465] #R,T,S
  energy_const=(197.3269)**2/(2*939.565)*(2*np.pi/L)**2 #hbar^2/2m in MeV fm^2
  
  # setup shared data
  dim1B = len(states)
  print("Number of SP states:", dim1B)
  #dim2B = dim1B * dim1B
  # basis definitions
  #bas1B     = range(dim1B)
  bas2B,subset2B,block_sizes,subset_sizes,bas_block2B ,mom_to_blk_nums= construct_basis_2B(full_state,states)
  
  print("Number of blocks in bas2B: ", len(bas_block2B))
  
  full.append(full_state)
  bass1.append(bas_block2B)
  bass.append(subset2B)
  
  Particle_P, Hole_H, Particle_A, Hole_A = pair_selector(bas_block2B, particles, holes)
  
  HH_PP_Filter, Hole_A_Filter = ph_filtering(Hole_H, Particle_P, Hole_A)
  #left_connections,right_connections = subset_selector(subset2B)
  bs_len = range(len(block_sizes)) #:O
  #basph2B   = construct_basis_ph2B(holes, particles)
  
  idp, idp_b = special_index(bas_block2B)
  
  #Initialize OP_Map with fixed variables idp, idp_b
  OP_Map_B.idp_b = idp_b
  OP_Map.idp_b = idp_b
  OP_Map.idp = idp

  fermi_index = particles[0]# so we know who's a hole, and who's not
  
  #fermi_understanding(bas_block2B, fermi_index)
  
  start_time = time.time()
  
  basph_block2B = ph_block_constructor(subset2B, full_state, fermi_index, len(bas2B), dim1B)
  
  print("Time Taken to obtain ph_blocks--- %s Minutes ---" % (round((time.time() - start_time)/60,4)))
  
  #print(bas_block2B[1])
  
  #print("Running Tests on basph_block2B!")
  
  #test_on_ph_refs(basph_block2B, full_state)#run simple tests
  
  #print("Ended Tests on basph_block2B")

  #print("zero", bas_block2B[17])
  

  # occupation number matrices
  occ1B     = construct_occupation_1B(dim1B, holes, particles)
  occB_2B   = construct_occupationB_2B(block_sizes, bas_block2B, occ1B, bs_len)
  occC_2B   = construct_occupationC_2B(block_sizes, bas_block2B, occ1B, bs_len)
  occphA_2B = construct_occupationA_2B(basph_block2B, occ1B)#block matrices for ph trans
  
  np.set_printoptions(linewidth=np.inf)
  
#  for blks in occphA_2B:
#    print(blks)

  
  bn2=Bernoulli_generator(8)# Go up to 3 terms in expansion
  bn2[1]=-0.5#follow 2nd convention for Bernouli numbers   
  
  # store shared data in a dictionary, so we can avoid passing the basis
  # lookups etc. as separate parameters all the time
  user_data  = {
    "dim1B":      dim1B, 
    "holes":      holes, 
    "particles":  particles,
    "subset2B":   subset2B,
    "bas_block2B": bas_block2B,
    "basph_block2B": basph_block2B,
    "subset_sizes": subset_sizes,
    "block_sizes": block_sizes,
    "Particle_P": Particle_P,
    "Hole_H":     Hole_H,
    "Hole_A":     Hole_A,
    "HH_PP_Filter": HH_PP_Filter, 
    "Hole_A_Filter": Hole_A_Filter,   
    "bs_len":     bs_len,
    #"basph2B":    basph2B,
    "idp":        idp,
    "idp_b":      idp_b,
    #"idxph2B":    idxph2B,
    "occ1B":      occ1B,
    "occB_2B":    occB_2B,
    "occC_2B":    occC_2B,
    "occphA_2B":  occphA_2B,
    "bn2": bn2,
    # variables for sharing data between ODE solver
    "eta_norm":   0.0,                #Initial norm of eta
    "dE":         0.0,                # Initial dOmega/ds
    "calc_eta":   eta_white,          # specify the generator (function object)
    "calc_rhs":   RHS_Cal,         # specify the right-hand side and truncation
    "special_commutator": special_commutator, #commutator used to evaluate 1-2 body commutators
    "states": states,
    #"OP_Map": OP_Map
  }

  global_user_data.append(user_data) # for use in other scripts
  # set up initial Hamiltonian
  H1B,H2B=Inf_Matter_Ham(full_state, energy_const, Combined_consts, k_alpha, degen, L, user_data)
  #print(OP_Map_Test(bas2B, block_sizes, bas_block2B, idp, H2B))#ensure that OP_Map is doing what it needs

  E, f, Gamma = normal_order(H1B, H2B, user_data)#Normal ordered Hamiltonian with Hartree Fock energy
  HH.append(H2B)
  initializations.append([E,f, Gamma])
  
  #print(fully.append(Gamma) )
  #eigs.extend(np.linalg.eig(Gamma)[0])#store eigenvalues of H2B. This is interesting
  
  # Append Initial Hamiltonian to dictionary since that won't be changing in the Magnus expansion
  user_data["E"]= E
  user_data["f"]= f
  user_data["Gamma"]= Gamma
  
  #print(subset2B[7])
  #print(list(reverse_enumerate(subset2B[7],subset_sizes[7])))
  
  #Initializations
  #Initial_Omega0=0.
  #Initial_Omega1=np.zeros((dim1B,dim1B))
  Initial_Omega2=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
  #Transformed_Hamiltonian = Transformed_Ham(Initial_Omega1, Initial_Omega2, user_data)
  Transformed_Hamiltonian = E, f, Gamma #initial transformation preserves H
  
  #time_test_for_map(Gamma, idp_b, subset2B, full_state, mom_to_blk_nums, idp)
  
  #eta_1B, eta_2B = user_data["calc_eta"](Transformed_Hamiltonian[1], Transformed_Hamiltonian[2], user_data)
  
  #user_data["eta_norm"] = np.linalg.norm(eta_1B, ord='fro') + calc_full2B_norm(eta_2B, subset2B,bs_len,idp_b)
  
  # reshape generator into a linear array (initial ODE vector)
  y0   = List_reshape(Initial_Omega2,block_sizes, bs_len)
  Omega_F= y0# flattened Omega
  
  # integrate flow equations 
  sinitial=0
  sfinal = 50  
  ds = 1
  num_points = (sfinal-sinitial)/ds +1
  flow_pars = np.linspace(sinitial,sfinal,int(num_points))
  print( "Reference Energy (MeV):", round(E,4))
  #print(eta_white(f,Gamma, user_data)[0])
  
  print ("%-14s   %-11s   %-14s   %-14s   %-14s  %-14s   %-14s   %-14s"%(
    "s", "E/A" , "DE(2)", "DE(3)", "E+DE", 
    "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print ("-" * 130)
  start_time = time.time()
  DE3=0.0
  
  
  for flow_p in flow_pars:
      
    #print("Omega",LA.norm(Omega_1B))
    #print("f",LA.norm(f-np.diag(np.diag(f))))
    energy.append(E)#append energy for plotting
    flow.append(flow_p)#append flow parameter for plotting


    DE2 = calc_mbpt2(f, Gamma, user_data)
    #DE3 = calc_mbpt3(f, Gamma, user_data)

    #norm_fod     = calc_fod_norm(f, user_data)
    norm_fod = 0.0 #from momentum conservation
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)
    
    ys = ds * derivative_wrapper(Omega_F, user_data, Transformed_Hamiltonian) + Omega_F

    print ("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
        flow_p, E/A , DE2, DE3, E+DE2+DE3, user_data["eta_norm"], norm_fod, norm_Gammaod))
    
    #ys = ds * derivative_wrapper(Omega_F, user_data, Transformed_Hamiltonian) + Omega_F

    Omega_2B = get_operator_from_y(ys, dim1B, block_sizes,bs_len)#get new Omegas
    
    #Omega_checker(Omega_2B, Hole_H, Particle_P, bs_len)
    
    worm(Omega_2B, bs_len, flow_p, countering)
    Transformed_Hamiltonian = Transformed_Ham(Omega_2B, user_data)#get  Hamiltonian to use MBPT, and print out values
    E, f , Gamma = Transformed_Hamiltonian #get  Hamiltonian to use MBPT, and print out values
    
    if (abs(DE2/E) < 10e-8): 
        print("Time Taken to Run--- %s Minutes ---" % (round((time.time() - start_time)/60,4)))
        print( "Correlation Energy (MeV):", round(E-user_data["E"],4))
        print("Energy per particle (MeV):", round(E/A, 4))
        results.append([E,f,Gamma,Omega_2B])
        print(Counter(countering))
        break
    Omega_F = List_reshape(Omega_2B,block_sizes, bs_len)
#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
main()

#cProfile.run("main()")     
#if __name__ == "__main__": 
#  main()
#plt.plot(flow, energy)
