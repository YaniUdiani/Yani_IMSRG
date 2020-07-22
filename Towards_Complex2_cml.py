#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 15 12:25:15 2020

@author: YaniUdiani
"""

""" NOTE TO SELF: GET RID OF 'mom_to_blk_nums' """
#------------------------------------------------------------------------------
""" 
This version combines Towards_Complex with J_Dev5.py

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
import cProfile
import array as arr

import argparse

#-----------------------------------------------------------------------------------
# Ploting variables
#-----------------------------------------------------------------------------------
flow=[] #list of flow parameters
energy=[]#list of energies for plotting
#Magic=[]
#HH=[]
#initializations = [] #used to store E_ref, and f_Hartree-fock
results = [] #used to store E_g.s, f,Gamma, Omega1B, Omega2B 
#global_user_data =[]
#bass = []
#bass1 = []
#full = []
#fully = []
#-----------------------------------------------------------------------------------
# Functions to compute Infinite Matter single particle states
#-----------------------------------------------------------------------------------

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

def pre_process(Magic_Numbers):
  
  """This func converts the string of magic numbers into a list of #'s"""
  
  numeric = []
  
  temp = Magic_Numbers.split(",")
  
  numeric.append(2)
  
  for i in range(1,199): #assume that there are at most 200 shells
    #I really don't see how I could ever be working with 200 shells
    
    numeric.append(int(temp[i]))
    
  numeric.append(23506)
  
  
  return numeric


def get_N_max_newshells(N_Max, A, degen, Magic_Numbers):
  
  """ This function figures out which shells the holes occupy, then uses
  N_Max to add the shells that will contain the particle states. It also returns
  
  Hole_Nmax for the fermi momentum. Note that right now, you can't arbitrarily 
  pick which states are holes and particles. I'm just going to organize them
  according to the norm of the momenta. 
  """
  
  if(degen == 2):
    
    if(A not in Magic_Numbers and A < max(Magic_Numbers)):
      
      
      print("""The number of particles you entered doesn't form a closed shell.
            Your number of particles must be one of these values: """, Magic_Numbers)
      
      sys.exit()
      # I'm assuming that I'll never get past A = 23506 = Magic_Numbers[199] (for PNM)
      #given the way things are going
      
    else: 
      
      Hole_Nmax = Magic_Numbers.index(A)
      
      Total_shells = Hole_Nmax + N_Max
      
      return full_state_gen_3D(Total_shells, degen), Hole_Nmax
    
    
  else:
    
    Magic_Numbers = [2 * x for x in Magic_Numbers] # protons double # of states
    
    
    if(A not in Magic_Numbers and A < max(Magic_Numbers)):
      
      
      print("""The number of particles you entered doesn't form a closed shell.
            Your number of particles must be one of these values: """, Magic_Numbers)
      
      sys.exit()
      # I'm assuming that I'll never get past A = 2 * 23506 = Magic_Numbers[199] (for PNM)
      #given the way things are going
      
    else: 
      
      Hole_Nmax = Magic_Numbers.index(A)
      
      Total_shells = Hole_Nmax + N_Max
      
      return full_state_gen_3D(Total_shells, degen), Hole_Nmax
    

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
    
    blk_nums_to_mom = {}# keys as block_num, values =  momenta+spin
    Subset=[]# List will contain lists of filterd pairs in a given block.
    #This list filters the pairs in each block. It acts as subset2B for each block    
    Bas_block2B=[[] for i in range(num_blocks)]# Bas2B with each block clearly demarcated
    block_num = 0
    
    for keys in block:
      
        blk_nums_to_mom[block_num] = keys
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
    return Bas2B, Subset, block_sizes, subset_sizes, Bas_block2B, blk_nums_to_mom


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

    
#
# We use dictionaries for the reverse lookup of state indices
#

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
      #basph_blocks2B.append(ph_arrangement + temp)
      
      basph_blocks2B.append([ph_arrangement,temp])#p-p and h-h sector will be at basph_blocks2B[1]
      
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
                 
      for (i,j) in basph_blocks2B[block_num1][1]:
        
        for (k,l) in basph_blocks2B[block_num2][1]:
          
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
def construct_occupationA_2B(basph_block2B, occ1B, occ_DTYPE):
  
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
  mirrored terms. Also, I will only be computing the matrix for the ph sector
  in each block to further simplify my calculations.
  """
  
  occ=[]#initialize list of block matrices for ph trans

  for blocks in basph_block2B:# go through blocks
    
    temp_occ_ref = np.zeros((len(blocks[0]), len(blocks[0])), dtype = occ_DTYPE)
      
    for i1, (i,j) in enumerate(blocks[0]):#just get elements for ph sector
      
      temp_occ_ref[i1, i1] = occ1B[i] - occ1B[j]
      
    occ.append(temp_occ_ref)

  return occ


def ph_trans_then_mult(x, y, resultant, ph_block, block_occph, idp, sign, DTYPE):
  
  """ See description in ph_trans_prep()
  Note that ph_block[0] = ph/hp sector and ph_block[1] = pp + hh = E (everything else sector)
  """
  
  #Perform PH Transform 
  
  zero = len(ph_block[0]); one = len(ph_block[1])
  
  x_ph_ph = np.zeros((zero, zero), dtype = DTYPE)  
  y_ph_ph = np.zeros((zero, zero), dtype = DTYPE)
  
  x_E_ph = np.zeros((one, zero), dtype = DTYPE) 
  y_ph_E = np.zeros((zero, one), dtype = DTYPE)
  
  
  
  for i1, (a,b) in enumerate(ph_block[0]):
    
    for i2, (c,d) in enumerate(ph_block[0]): #obtain x_ph_ph and y_ph_ph
      
      block_num, index_1 = idp[(a,d)]
      
      index_2 = idp[(c,b)][1]
      
      x_ph_ph[ i1, i2 ] -= x[block_num][index_1, index_2]
      
      y_ph_ph[ i1, i2 ] -= y[block_num][index_1, index_2]      
      
      
      
  for i2, (c,d) in enumerate(ph_block[0]): #obtain x_E_ph      
    
    for i1, (a,b) in enumerate(ph_block[1]):#ph_block[0] has ph terms, 
      #ph_block[1] has everything else
    
      
      block_num, index_1 = idp[(a,d)]
      
      index_2 = idp[(c,b)][1]
      
      x_E_ph[ i1, i2 ] -= x[block_num][index_1, index_2]
      #x_E_ph[ i1, i2 ] -= x_map[[(a,d), (c,b)]]
      
      y_ph_E[ i2, i1 ] -= y[block_num][index_1, index_2] 
      #really should be y_ph_E[i2,i1] -= y[block_num][idp[(d,a)], idp[(b,c)][1]] 
      #(Note: See J_Dev2.py ph routine to see why this should be the case)
      #but matrix elements are anti-symmetrized!
      
      
  
 ###### Perform multiplications, then immediately inverse transform, 
 #so resulting prod matrix is cleared from memory (hopefully)
 
 ##################### ph-ph block mult & inverse ##################### 
     
  # multiply in ph-ph block of resulting prod matrix    
  prod_ph_ph = dot(x_ph_ph, dot(block_occph, y_ph_ph))
  
  #Inverse Transform ph-ph block of prod matrix
  
  for i1, (a,b) in enumerate(ph_block[0]): 
    
    for i2, (c,d) in enumerate(ph_block[0]):
      
      block_num, index_1 = idp[(a,d)];    
      
      resultant[block_num][index_1, idp[(c,b)][1]] -= prod_ph_ph[ i1, i2 ]
      
      resultant[block_num][idp[(b,c)][1], idp[(d,a)][1]] = sign * -prod_ph_ph[ i1, i2 ] #conjugate term             
      
      # mirror contribute to hermitian conjugate of resultant_map
  
  ##################### ph-E and E-ph block mult & inverse ##################### 
   
  # multiply in ph-E block  of resulting prod matrix
  prod_ph_E = dot(x_ph_ph, dot(block_occph, y_ph_E))# multiply in ph-E sector
  
   
  # multiply in ph-E block  of resulting prod matrix
  prod_E_ph = dot(x_E_ph, dot(block_occph, y_ph_ph))# multiply in ph-E sector
  
    #Inverse Transform ph-E block of prod matrix
  
  for i1, (a,b) in enumerate(ph_block[0]): 
    
    for i2, (c,d) in enumerate(ph_block[1]):
      
      block_num, index_1 = idp[(a,d)]; 
      
      index_2 = idp[(c,b)][1]
      
      flip_1 = idp[(d,a)][1]; flip_2 = idp[(b,c)][1]
      
      resultant[block_num][index_1, index_2] -= prod_ph_E[ i1, i2 ]
      
      resultant[block_num][flip_2, flip_1] = sign * -prod_ph_E[ i1, i2 ] #conjugate term       
      
      # mirror contribute to hermitian conjugate of resultant
      
        
      #Get results in E_ph sector of prod matrix:
      
      resultant[block_num][flip_1, flip_2] -= prod_E_ph[ i2, i1 ]#note that resultant has no definite symm
      
      resultant[block_num][index_2, index_1] = sign * -prod_E_ph[ i2, i1 ] #conjugate term 
  
  ##################### ph-ph block mult & inverse ##################### 
      
  # multiply in E-E block of resulting prod matrix
  prod_E_E = dot(x_E_ph, dot(block_occph, y_ph_E))# multiply in ph-ph sector
  
  
  for i1, (a,b) in enumerate(ph_block[1]): 
    
    for i2, (c,d) in enumerate(ph_block[1]):
      
      block_num, index_1 = idp[(a,d)]; 
      
      resultant[block_num][index_1, idp[(c,b)][1]] -= prod_E_E[ i1, i2 ]
      
      resultant[block_num][idp[(b,c)][1], idp[(d,a)][1]] = sign * -prod_E_E[ i1, i2 ] #conjugate term       
        
      
  return      

def ph_trans_prep(x, y, basph_block2B, occphA_2B, bs_len, sign, idp, block_sizes, DTYPE):
  
  """ Takes in block operators x, y and preps them for conversion to ph representation. 
  Since both operators will have the same block structure in ph representation,
  their non-zero blocks are obtained using the blocks of basph_block2B.
  
  For each block of idxph2B, the function ph_trans_then_mult() obtains both x_ph[ph_block_num]
  and y_ph[ph_block_num], then multiplies them with the according occupation matrix.
  
  ph_trans_then_mult() then does the inverse transform to obtain resultant
  block matrix.
  
  """
  resultant = [np.zeros( (block_sizes[i], block_sizes[i]), dtype = DTYPE ) for i in bs_len]#initialize list of block matrices
  
  #resultant_map = OP_Map("R")
  #resultant_map.obj = resultant
  
  for ph_block_num, ph_block in enumerate(basph_block2B):
    
    ph_trans_then_mult(x, y, resultant, ph_block, occphA_2B[ph_block_num], idp, sign, DTYPE)
    
  return resultant



        
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
  DTYPE = user_data["DTYPE"]
  
  norm = 0.0
  
    
  
  if DTYPE == np.cdouble:
    
    for block_num in HH_PP_Filter:# go through blocks 
              
      for ket in Hole_H[block_num]:#no time will be wasted looping over Hole_H[blk#]
        
        i2 = idp_b[block_num][ket]
        
        for bra in Particle_P[block_num] :#if Particle_P[block_num] is empty,
            
          #i1 = idp_b[block_num][bra]
          val = Gamma[block_num][idp_b[block_num][bra], i2]
          
          norm += 8 * np.conj(val) * val #python returns complex type, even though should be real
  
    return np.sqrt(norm.real)#so just take real component
      
  else:
    

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

def calc_full2B_norm(Gamma, bs_len):#compare timing with other calc_full2B_norm()
    
  #Gamma_Map = OP_Map_B("Gamma")#create instance of OP_Map_B
  #Gamma_Map.obj = Gamma #create class operator that it sees
  
  norm = 0.0
  
  for block_num in bs_len:# go through blocks
    #find pairs in block     
    norm  += LA.norm(Gamma[block_num])**2
#    for i1, bra in enumerate(subset2B[block_num]):    
#      for i2, ket in enumerate(subset2B[block_num]):
#        
#          norm += 4 * Gamma[block_num][ i1, i2]**2  

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
def construct_occupation_1B(dim1B, holes, particles, occ_DTYPE):

  occ = np.zeros(dim1B, occ_DTYPE)

  for i in holes:
    occ[i] = 1.

  return occ

# diagonal matrix: 1 - n_a - n_b
def construct_occupationB_2B(block_sizes, bas_block2B, occ1B, bs_len, occ_DTYPE):
  
  occ=[np.zeros((block_sizes[i],block_sizes[i]), occ_DTYPE) for i in bs_len]#initialize list of block matrices

  for block_num in bs_len:# go through blocks
    for i1, (i,j) in enumerate(bas_block2B[block_num]):
      occ[block_num][i1, i1] = 1. - occ1B[i] - occ1B[j]

  return occ

# diagonal matrix: n_a * n_b 
def construct_occupationC_2B(block_sizes, bas_block2B, occ1B, bs_len, occ_DTYPE):
  
  occ=[np.zeros((block_sizes[i],block_sizes[i]), occ_DTYPE) for i in bs_len]#initialize list of block matrices

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
  DTYPE = user_data["DTYPE"]
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
  eta2B = [np.zeros((block_sizes[i],block_sizes[i]), dtype = DTYPE) for i in bs_len]#initialize list of block matrices
  

  for block_num in HH_PP_Filter:# go through blocks
    
    disp = subset_size[block_num]# how far away to the right is (b,a) from (a,b)? same for (i,j)

    for (i,j) in Hole_H[block_num]:
      
      
      i2 = idp_b[block_num][(i,j)]#location of pair (i,j)    
      
      for (a,b) in Particle_P[block_num]:
        
        i1 = idp_b[block_num][(a,b)]#locaion of pair (a,b)
      
        
        block_num1, index1 = idp[(a,i)]
        block_num2, index2 = idp[(a,j)]
        block_num3, index3 = idp[(b,i)]
        block_num4, index4 = idp[(b,j)]
        
#          if(block_num1 == block_num4):
#            print([(a,i),(b,j), block_num1])
        
        denom = ( 
        f[a] + f[b] - f[i] - f[j]  
        
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
    
    DTYPE = user_data["DTYPE"]
               
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

        xyi = ph_trans_prep(x, y, basph_block2B, occphA_2B, bs_len, sign, idp, block_sizes, DTYPE)
        
        Output_2B = List_operation(bracket, 0.5, block_sizes,sign,"Scalar_Mul", bs_len)
        
        if(sign == 1): #if commutator coming from Hamiltonian routine
          
            dim1B     = user_data["dim1B"]
            idp_b    = user_data["idp_b"]
            #Initializations
            if DTYPE == np.cdouble:
              
                Output_0B = 0.0 + 0.0j #seems like E will be complex
                
            else:
              
              Output_0B = 0.0
              
            Output_1B = np.zeros((dim1B, ), dtype = DTYPE)
            
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
                    
                    Output_1B[p] += xy[block_num][i1, i1] 
                
#                for (r,p) in subset2B[block_num]:
#                  
#                    Output_1B[p, p] +=  xyz_map[[(r,p), (r,p), block_num]]
#                    
#                    Output_1B[r, r] += xyz_map[[(p,r), (p,r),block_num]] 
                    
                    
            for block_num in bs_len:# go through blocks    
                
                temp_blk = subset2B[block_num]
                
                disp = subset_size[block_num] # = to how many pairs away (j,i) is from (i,j)      
                
                #Hermitian = copy.copy(subset2B[block_num]) 
                
                ranger = list(range(len(temp_blk)))
                
                for i1, (r,p) in reverse_enumerate(temp_blk,disp):
                  
                    #index = idp_b[block_num][(r,p)]#locaion of pair (r,p)
                                      
                    Output_1B[p] +=  xyz[block_num][i1, i1]
                    
                    Output_1B[r] +=  xyz[block_num][i1 + disp, i1 + disp]
                    
                    #for i2, (j,q) in enumerate(temp_blk):
                    for i2 in ranger:  
    
                            
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
                        
                    del ranger[i1] 
                    #delete (r,p) so it does show up in Hermitian since transpose is manually coded                        

            return Output_0B, Output_1B, Output_2B
          
          
        else: #Omega_0B and Omega_1B are zeros, so only worry about 2BD piece in RHS routine
          
            for block_num in bs_len:# go through blocks
                
                disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)                 
                #Hermitian = copy.copy(subset2B[block_num]) 
                
                #for i1, (r,p) in reverse_enumerate(Hermitian,disp):
                    #for i2, (j,q) in enumerate(Hermitian):
                ranger = list(range(len(subset2B[block_num])))
                
                for i1 in reversed(ranger):
                  
                    for i2 in ranger:
                      
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
                        
                    #del Hermitian[i1] 
                    del ranger[i1] 
                    #delete (r,p) so it does show up in Hermitian since transpose is manually coded    
                    
            return Output_2B   
      
      
    
                    
    if(type(x) == list and type(y) == np.ndarray):#2B-1B 


    ########## Note: The 1-BD piece from this commutator is

    # Output_1B[i,j]-= y[i,j] * (x[[(a,i),(b,j)]] - x[(b,i),(a,j)]) = 0
    
    # x = Omega2B, y = f or some diagonal matrix from commutators in bch routine    

    # Since i = j, and k_a + k_i = k_b_+ k_j, then a = b           
          
        subset_size = user_data["subset_sizes"]
        block_sizes = user_data["block_sizes"]
        bs_len = user_data["bs_len"]
        subset2B = user_data["subset2B"]
        
        
        Output_2B=[np.zeros((block_sizes[i],block_sizes[i]), DTYPE) for i in bs_len]#initialize list of block matrices
        
                    
        for block_num in bs_len:# go through blocks  
          
            disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)    
            Hermitian = copy.copy(subset2B[block_num])  
            
            for i1, (i,j) in reverse_enumerate(Hermitian,disp):
                for i2, (k,l) in enumerate(Hermitian):                  
                    
                    
                    mval = -y[l] * x[block_num][i1, i2 + disp] #I hope this is stored in cache
                    
                    mval += (y[k] - y[i]) * x[block_num][i1, i2]
                    
                    mval += y[j] * x[block_num][i1 + disp, i2]
                    
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
    #subset2B = user_data["subset2B"]
    bs_len = user_data["bs_len"]
    #idp = user_data["idp"]
    #idp_b    = user_data["idp_b"]
    DTYPE = user_data["DTYPE"]
    
    #Zero_Blocks = user_data["Zero_Blocks"]

    RHS_2B=[np.zeros((block_sizes[i],block_sizes[i]), DTYPE) for i in bs_len]#initialize RHS of diffy q

    for n in range(len(bn2)):
      
        if n==0: 
          
            nth_2B = Eta_2B #Initial commutators defined to be Eta


        else:
            #nth_1B will always be zero, see latex document [\eta^{2BD}(0),\,\,\eta^{2BD}(ds)]^{1BD}=0

            nth_2B=special_commutator(Omega_2B,nth_2B, user_data, -1)#store two body-two body commutator
            #; -1 for anti-hermitian 2B piece

            #Omega_checker(nth_2B, bs_len, Zero_Blocks)#check which new blocks became non-zero
            
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        
        shiftymatrix_2B=List_operation(nth_2B, 1/np.math.factorial(n), block_sizes,"NA","Scalar_Mul", bs_len)
        
#        if (calc_full2B_norm(shiftymatrix_2B, subset2B,bs_len,idp_b) < 1e-6 ):
#
#            break        
        #RHS_1B+=bn2[n]*shiftymatrix_1B
        
        First_mul = List_operation(shiftymatrix_2B, bn2[n], block_sizes,"NA","Scalar_Mul", bs_len)
        RHS_2B = List_operation(RHS_2B, First_mul, block_sizes,+1,"List_Add", bs_len)
        
        if (calc_full2B_norm(shiftymatrix_2B, bs_len) < 1e-6 ):

            break   
          
    return  RHS_2B

def Transformed_Ham(Omega_2B, user_data):# Generates new Hamiltonian by performing BCH expansion
  
    bn2 = user_data["bn2"]
    block_sizes = user_data["block_sizes"]
    #subset2B = user_data["subset2B"]
    E=user_data["E"]
    f=user_data["f"]
    Gamma=user_data["Gamma"]
    bs_len = user_data["bs_len"]
    DTYPE = user_data["DTYPE"]
    #idp = user_data["idp"]
    #idp_b    = user_data["idp_b"]
    
    H_0B = 0. #Initialize Hamilitonian
    H_1B = np.zeros_like(f)
    H_2B = [np.zeros((block_sizes[i],block_sizes[i]), DTYPE) for i in bs_len]#initialize list of block matrices

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
        if (abs(shiftymatrix_0B) < 1e-4 ):#MeV
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
  #subset2B = user_data["subset2B"]
  block_sizes = user_data["block_sizes"]
  bs_len = user_data["bs_len"]
  #idp = user_data["idp"]
  #idp_b    = user_data["idp_b"]
  
  # extract operator pieces from solution vector
  Omega_2B = get_operator_from_y(y, dim1B, block_sizes, bs_len)

  # calculate the 2BD generator
  Eta_2B = calc_eta(Transformed_Hamiltonian[1], Transformed_Hamiltonian[2], user_data)
  
  # share data
  #user_data["dE"] = dOmega_0B #storing dOmega_0B/ds in dE
  user_data["eta_norm"] =  calc_full2B_norm(Eta_2B, bs_len)  
  
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
    #bas_block2B = user_data["bas_block2B"]
    bs_len = user_data["bs_len"]
    dim1B = user_data["dim1B"]
    idp_b = user_data["idp_b"]
    subset_size = user_data["subset_sizes"]
    DTYPE = user_data["DTYPE"]
    
    
    H1B = np.zeros((dim1B, ), dtype = DTYPE)# store as vector since diagonal
    
    for i in states:
      
        H1B[i] = energy_const * np.dot(full_state[i][0:3], full_state[i][0:3]) #0:3 doesn't include 3
        
    H2B=[np.zeros((block_sizes[i],block_sizes[i]), DTYPE) for i in bs_len]#initialize list of block matrices
    
    for blocky in bs_len:# fill in all blocks
      
        disp = subset_size[blocky]
        
        Hermitian = copy.copy(subset2B[blocky])# I plan to exploit H2B's hermiticity: 
        #Once a pair (p,q) has been considered in the bra, then it won't be considered 
        #in the ket via my loop. I will manually compute the transpose.
        for (p,q) in subset2B[blocky]: #I could use enumerate 
          
            #Get respective indices of pairs in each block
            block_loc_pq = idp_b[blocky][(p,q)]
            #block_loc_pq=bas_block2B[blocky].index((p,q))
            block_loc_qp = block_loc_pq + disp
            #block_loc_qp=bas_block2B[blocky].index((q,p))
          
            for (r,s) in Hermitian:

                
                block_loc_rs = idp_b[blocky][(r,s)]
                #block_loc_rs=bas_block2B[blocky].index((r,s))

                block_loc_sr = block_loc_rs + disp
                #block_loc_sr=bas_block2B[blocky].index((s,r))
                
                
#                if (np.all( full_state[p][0:3] + full_state[q][0:3] != full_state[r][0:3] + full_state[s][0:3] )):
#                    print("CM Momentum is not being conserved!")
                #ensure that center of mass momentum is conserved for all matrix elements
                
                #Compute anti-symmetrized matrix element <(p,q)| H2B |(r,s)>_AS
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
    
    E += H1B[i]
    
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
     
     f[p] += H2B[block_num][i1, i1]  #since (i,p) and (i,q) are in the same 
     #Blk, p = q

  # 2B piece of new normal ordered Hamiltonian is H2B. No need for corrections to H2B

  return E, f, H2B

#def Cheap_Self_Energy(E, Gamma, f, particles, dim1B):
#  Sigma = np.zeros (dim1B, dim1B)
#-----------------------------------------------------------------------------------
# Perturbation theory
#-----------------------------------------------------------------------------------
def calc_mbpt2(f, Gamma, user_data):
  
  #bs_len = user_data["bs_len"]
  #idp    = user_data["idp"]
  idp_b    = user_data["idp_b"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  HH_PP_Filter = user_data["HH_PP_Filter"] 
  
  DTYPE = user_data["DTYPE"]
  
  if DTYPE == np.cdouble:
    
    DE2 = 0.0 + 0.0j #seems like DE will be complex
      
  else:
    
    DE2 = 0.0  
  
  for block_num in HH_PP_Filter:# go through blocks with HH and PP terms
      
    for (i,j) in Hole_H[block_num]:
    
      index = idp_b[block_num][(i,j)]# I can reuse this
      
      for (a,b) in Particle_P[block_num]:
      
        
        denom = f[i] + f[j] - f[a] - f[b]
        me    = Gamma[block_num][idp_b[block_num][(a,b)], index]
        DE2  += np.conj(me) * me/denom #note that np.conj(me) * me will return complex num

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
  
  
def Omega_checker(Omega, bs_len, Zero_Blocks):
   
   
  for block_num in bs_len:
     
    if(LA.norm(Omega[block_num]) != 0.0):
         
      if(block_num in Zero_Blocks):
        
        del Zero_Blocks[Zero_Blocks.index(block_num)]
        
        
  return
        
#    if( Hole_H[block_num] and Particle_P[block_num]):
#       
#         
#      print(LA.norm(Omega[block_num]))        
#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------
  

#def main(A, N_Max, rho, degen):

def main(args): 
  
  results.clear()#If I'm making many calls to this script's main
  #then I don't want results to keep ballooning up
  
  #Magic.clear()
  
  print(" ")
  print("This code uses precalculated magic numbers to determine Nmax for a given number of hole states")
  
  print("Magic_Numbers.txt must be in same directory as this file")
  
  f = open("Magic_Numbers.txt", "r")
  
  Magic_Numbers = f.read() #Magic numbers from N_max = 0 : N_max = 200
  #Magic_Numbers[0] is 0th shell  
  
  Magic_Numbers = pre_process(Magic_Numbers)# convert them into list of ints
    
  #Magic.append(Magic_Numbers)
  
  
  parser = argparse.ArgumentParser(description='Input arguments: --A=... --Nmax=... --den=... --g=... ')
  
  parser.add_argument("--A", required=True, type=int, 
  help="Number of particles in box. Only number of holes in closed shells are allowed",
  choices = Magic_Numbers)
  
  parser.add_argument("--Nmax", required=True, type=int,
                      help="This specifies the model space")
  
  parser.add_argument("--den", required=True, type=float, 
                      help="Density of particles in box")
  
  parser.add_argument("--g", required=False, type=int, choices=[2,4],
                      default= 2, help="=4 for SNM, & =2 for PNM")
  
  parser.add_argument("--comp", required=False, type= int, default=0, choices=[0,1],
  help="pick 0 (np.double) for real potential (pot) or 1 (np.cdouble) for complex pot")  
  
  #parser.add_argument("--DE", required=False, type=str, default="off",
  #                   choices=["on","off","On","Off","ON","OFF"], help="MBPT(3) on or off?")
  parser.add_argument("--DE", required=False, type=int, default=0,
                      choices=[0,1], help="0=MBPT(3) off and 1=MBPT(3) on")
  
  parser.add_argument("--T", required=False, type=int, choices=range(1,17), default=6, 
                      help="BCH expansion will go up to T terms") 
  
  parser.add_argument("--step", required=False, type=float, default = 1.0,
                      help="Step size in s: 1.0 seems to work fine")  
  
  args = parser.parse_args(args)
  
  
  
  A = args.A #specify number of particles you want to include
  
  N_Max = args.Nmax#How many shells above the holes you want to include. 
  
  #Unlike before, N_max cannot equal 0. We need particle states!
  
  rho = args.den #Define density in fm^-3  
  
  degen = args.g #If degen=4, then the matter is symmetric and isospin 
  #projections are considered. If degen=2, there is only one type of particle, 
  #so isospin projections are not considered. Degen is only either 2 or 4.
  
  DE_flag = args.DE#Current 3rd order implementation  rightnow is 
  #realllly slow and should not be used, but here it is. Also it 
  #might not work with complex numbers. Don't really care to look into that rn.
  
  trunc = args.T #Set max # of terms in BCH and Magnus expansion
  
  complex_def = args.comp#determine if potential is real or complex
  
  ds = args.step #obtain step size if requested
  
  ####Can change N_Max (determines # of particles),rho (determines size of box and Fermi lvl), and degen (type of matter) ######
  
  #A = 14 #specify number of particles you want to include
    
  #N_Max = 1 #How many shells above the holes you want to include. Unlike before,
  
  #N_max cannot equal 0. We need particle states!
  
  
  #rho = 0.2 #Define density in fm^-3  
  
  #degen = 2#If degen=4, then the matter is symmetric and isospin projections are considered. If degen=2, there is only
  #one type of particle, so isospin projections are not considered. Degen is only either 2 or 4. 

  temp_full_state, hole_Nmax = get_N_max_newshells(N_Max, A, degen, Magic_Numbers)
  

  #############################

  full_state = sorter(temp_full_state, hole_Nmax)#order the states so that holes come before particle states in the list. I like it that way
  
  full_state = np.array(full_state)#covert it to an array for easy viewing
  #Note that full state corresponds to single particle states labled by mode #'s n, not wavenumbers k

  #print(full_state)
  
  #fully.append(full_state)
  
  holes, particles = One_BD_States(hole_Nmax, full_state)
  
  states = holes + particles
  
  if (A !=len(holes)):
    
    print("The number of holes aren't what they should be. Something went wrong.")  
    
  L=(A/rho)**(1/3)
  #L = 4.1212852998079734
  print(" ")
  
  print("Box Size (fm): ", L)
  
  print("Number of particles: ", A)
  
  Combined_consts=[(200/L**3)*(np.pi/1.487)**(3/2), -(178/L**3)*(np.pi/0.639)**(3/2), -(91.85/L**3)*(np.pi/0.465)**(3/2)]
  #Combined constants for matrix element. 
  
  k_alpha=[1.487,0.639,0.465] #R,T,S
  
  energy_const = (197.3269804)**2/(2*939.56542052)*(2*np.pi/L)**2 #hbar^2/2m in MeV fm^2

  #energy_const = 20.721155315 *(2*np.pi/L)**2 #hbar^2/2m in MeV fm^2
  
  # setup shared data
  dim1B = len(states)
  print("Number of SP states:", dim1B)
  #dim2B = dim1B * dim1B
  # basis definitions
  #bas1B     = range(dim1B)
  bas2B,subset2B,block_sizes,subset_sizes,bas_block2B, blk_nums_to_mom = construct_basis_2B(full_state,states)
  
  print("Number of blocks in bas2B: ", len(bas_block2B))
  
  print("Very crude estimate for memory demand of program (GB): ", 
      round(( np.sum(np.array(block_sizes)**2)* 8 * ( 11 + 5)  + 8*4*dim1B)  /1e9, 4 ))#*8 for sizeof(double), 
  #+11 for crudely counting number of 2BD block matrices, + 5 for guess for 
  #other overhead from objects that are not matrices, and +4 *dim1B* sizeof(double) for number of 1BD arrays  
  
  print(" ")
  
  Particle_P, Hole_H, Particle_A, Hole_A = pair_selector(bas_block2B, particles, holes)
  
  HH_PP_Filter, Hole_A_Filter = ph_filtering(Hole_H, Particle_P, Hole_A)
  #left_connections,right_connections = subset_selector(subset2B)
  bs_len = range(len(block_sizes)) #:O
  #basph2B   = construct_basis_ph2B(holes, particles)
  
  idp, idp_b = special_index(bas_block2B)
  
  #Initialize OP_Map with fixed variables idp, idp_b
  #OP_Map_B.idp_b = idp_b
  #OP_Map.idp_b = idp_b
  #OP_Map.idp = idp

  fermi_index = particles[0]# so we know who's a hole, and who's not
  
  start_time = time.time()
  
  basph_block2B = ph_block_constructor(subset2B, full_state, fermi_index, len(bas2B), dim1B)
  
  flag = True
  for blockss in basph_block2B:
    
    if(len(blockss[0]) == 0 or len(blockss[1]) == 0):
     
      flag = False
      
  if(flag == False):
    
    print("Some ph blocks are zeros")
      
  print(" ")
  print("Time Taken to obtain ph_blocks--- %s Minutes ---" % (round((time.time() - start_time)/60,4)))
  #print(basph_block2B)
  
  #print(bas_block2B[1])
  
  #print("Running Tests on basph_block2B!")
  
  #test_on_ph_refs(basph_block2B, full_state)#run simple tests
  
  #print("Ended Tests on basph_block2B")

  
  occ_DTYPE = np.double # occupation matrices are real
  
  if complex_def == 1 :
    
    DTYPE = np.cdouble # Seems like f, Gamma, Omega2B, 
    #eta2B and even E and DE2 are complex
    
  else:
    
    DTYPE = np.double # Seems like f, Gamma, Omega2B, 
    #eta2B and even E and DE2 are complex

  # occupation number matrices
  occ1B     = construct_occupation_1B(dim1B, holes, particles, occ_DTYPE)
  occB_2B   = construct_occupationB_2B(block_sizes, bas_block2B, occ1B, bs_len, occ_DTYPE)
  occC_2B   = construct_occupationC_2B(block_sizes, bas_block2B, occ1B, bs_len, occ_DTYPE)
  occphA_2B = construct_occupationA_2B(basph_block2B, occ1B, occ_DTYPE)#block matrices for ph trans
  
  np.set_printoptions(linewidth=np.inf)
  
  Zero_Blocks = list(bs_len)
  
  bn2=Bernoulli_generator(trunc)# Go up to 6 terms in expansion by default
  bn2[1]=-0.5#follow 2nd convention for Bernouli numbers   
  
  # store shared data in a dictionary, so we can avoid passing the basis
  # lookups etc. as separate parameters all the time
  user_data  = {
    "DTYPE": DTYPE,  
    "dim1B":      dim1B, 
    "holes":      holes, 
    "particles":  particles,
    "subset2B":   subset2B,
    "bas_block2B": bas_block2B,
    "basph_block2B": basph_block2B,
    "subset_sizes": arr.array("i", subset_sizes),
    "block_sizes": arr.array("i", block_sizes),
    "Particle_P": Particle_P,
    "Hole_H":     Hole_H,
    "Hole_A":     Hole_A,
    "HH_PP_Filter": arr.array("i", HH_PP_Filter), 
    "Hole_A_Filter": arr.array("i", Hole_A_Filter),   
    "bs_len":     arr.array("i",bs_len),
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
    "Zero_Blocks": Zero_Blocks
    #"OP_Map": OP_Map
  }

  #global_user_data.append(user_data) # for use in other scripts
  # set up initial Hamiltonian
  
  start_time = time.time()
  H1B, H2B=Inf_Matter_Ham(full_state, energy_const, Combined_consts, k_alpha, degen, L, user_data)
  
  print("Time Taken to obtain Minnesota Potential-- %s Minutes ---" % (round((time.time() - start_time)/60,4)))
  #print(OP_Map_Test(bas2B, block_sizes, bas_block2B, idp, H2B))#ensure that OP_Map is doing what it needs

  start_time = time.time()
  E, f, Gamma = normal_order(H1B, H2B, user_data)#Normal ordered Hamiltonian with Hartree Fock energy
 
  timer_normal_order = round((time.time() - start_time)/60,4)
    
  E = E.real
  
  #HH.append(H2B)
  #initializations.append([E,f, Gamma])
  
  #print(fully.append(Gamma) )
  #eigs.extend(np.linalg.eig(Gamma)[0])#store eigenvalues of H2B. This is interesting
  
  # Append Initial Hamiltonian to dictionary since that won't be changing in the Magnus expansion
  user_data["E"]= E
  user_data["f"]= f
  user_data["Gamma"]= Gamma
  
    
  Transformed_Hamiltonian = E, f, Gamma #initial transformation preserves H
  print("Time Taken to normal order potential-- %s Minutes ---" % (timer_normal_order))  

  # integrate flow equations 
  sinitial=0
  sfinal = 50  
  #ds = 1
  num_points = (sfinal-sinitial)/ds +1
  flow_pars = np.linspace(sinitial,sfinal,int(num_points))
  
  print(" ")
  print( "Reference Energy/A (MeV):", round(E/A,4))
  #print( "Reference Energy/A (MeV):", E/A)
  print(" ")

  print ("%-14s   %-11s   %-14s   %-14s   %-14s  %-14s   %-14s   %-14s"%(
    "s", "E/A" , "DE(2)", "DE(3)", "E+DE", 
    "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print ("-" * 126)
  
  DE3 = 0.0
  
  start_time = time.time()
  
  #print(type(H2B[22][0,0]))
  
  ############### Since Omega(ds) = eta(s = 0), No need to do commutator
  #routines, So I put the first part of the loop below.
  ###############
  
  calc_eta = user_data["calc_eta"]  
  
  Omega_2B = calc_eta(f, Gamma, user_data)#at s=ds, omega = eta
  
  Omega_checker(Omega_2B, bs_len, Zero_Blocks)#count # of non-zero blks in eta2B
  
  user_data["eta_norm"] =  calc_full2B_norm(Omega_2B, bs_len) 
  
  energy.append(E)#append energy for plotting
  flow.append(0.0)#append flow parameter for plotting


  DE2 = calc_mbpt2(f, Gamma, user_data).real
  
  if(DE_flag): #if user wants 3rd order on
    
    DE3 = calc_mbpt3(f, Gamma, user_data)

  #norm_fod     = calc_fod_norm(f, user_data)
  norm_fod = 0.0 #from momentum conservation
  norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)
  
  print ("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
      0.0, E/A , DE2, DE3, E+DE2+DE3, user_data["eta_norm"], norm_fod, norm_Gammaod))  
  
  Transformed_Hamiltonian = Transformed_Ham(Omega_2B, user_data)#get  Hamiltonian to use MBPT, and print out values
  
  E, f , Gamma = Transformed_Hamiltonian #get  Hamiltonian to use MBPT, and print out values
  
  E = E.real
  
  Omega_F = List_reshape(Omega_2B,block_sizes, bs_len)

  flow_pars = np.delete(flow_pars, 0) #delete s = 0 since I already plotted it.

  for flow_p in flow_pars:
      
    #print("Omega",LA.norm(Omega_1B))
    #print("f",LA.norm(f-np.diag(np.diag(f))))
    energy.append(E)#append energy for plotting
    flow.append(flow_p)#append flow parameter for plotting


    DE2 = calc_mbpt2(f, Gamma, user_data).real
    
    if(DE_flag): #if user wants 3rd order on
    
      DE3 = calc_mbpt3(f, Gamma, user_data)

    #norm_fod     = calc_fod_norm(f, user_data)
    #norm_fod = 0.0 #from momentum conservation
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)
    
    ys = ds * derivative_wrapper(Omega_F, user_data, Transformed_Hamiltonian) + Omega_F

    print ("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
        flow_p, E/A , DE2, DE3, E+DE2+DE3, user_data["eta_norm"], norm_fod, norm_Gammaod))
    
    #ys = ds * derivative_wrapper(Omega_F, user_data, Transformed_Hamiltonian) + Omega_F

    Omega_2B = get_operator_from_y(ys, dim1B, block_sizes,bs_len)#get new Omegas

    Omega_checker(Omega_2B, bs_len, Zero_Blocks)
    
    E_prev = E
    
    Transformed_Hamiltonian = Transformed_Ham(Omega_2B, user_data)#get  Hamiltonian to use MBPT, and print out values
    E, f , Gamma = Transformed_Hamiltonian #get  Hamiltonian to use MBPT, and print out values
    
    E = E.real
    
    #if (abs(DE2/E) < 10e-8):
    #if (user_data["eta_norm"] < 1e-2): 
    if ((E_prev - E)/A < 1e-3): 
      
        break
      
    Omega_F = List_reshape(Omega_2B,block_sizes, bs_len)
    
  
  ########### Plot last values since loop terminated ########### 
  
  #calc_eta = user_data["calc_eta"]  
  Eta_2B = calc_eta(f, Gamma, user_data)
  
  # share data
  #user_data["dE"] = dOmega_0B #storing dOmega_0B/ds in dE
  user_data["eta_norm"] =  calc_full2B_norm(Eta_2B, bs_len) 
    
  energy.append(E)#append energy for plotting
  flow.append(flow_p+1)#append flow parameter for plotting
  DE2 = calc_mbpt2(f, Gamma, user_data).real
  
  if(DE_flag): #if user wants 3rd order on
    
    DE3 = calc_mbpt3(f, Gamma, user_data)

  #norm_fod     = calc_fod_norm(f, user_data)
  #norm_fod = 0.0 #from momentum conservation
  norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)   
   
  print ("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
  flow_p+1, E/A , DE2, DE3, E+DE2+DE3, user_data["eta_norm"], norm_fod, norm_Gammaod))      
  
  print(" ")
  
  print("Time Taken to Run--- %s Minutes ---" % (round((time.time() - start_time)/60,4)))
  
  print(" ")
  
  print( "Correlation Energy/A (MeV):", round((E-user_data["E"])/A,4))
  
  print(" ")
  
  print("Energy per particle (MeV):", round(E/A, 4))
  
  print(" ")
  
  totty = len(bs_len)
  
  zzz = len(Zero_Blocks)
  
  frac = zzz / totty
  
  diff = totty - zzz
  
  print("Fraction of blocks of Omega that stayed zero", round(frac, 10),"=",
        zzz,"/", totty)
  
  print(" ")
  
  #results.append([E, f, Gamma, Omega_2B, frac, diff, dim1B, Zero_Blocks, blk_nums_to_mom])
  results.append([E, 0, 0 , Omega_2B, frac, diff, dim1B, Zero_Blocks, blk_nums_to_mom])#don't need f & Gamma right now 

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
#main(14, 1 , 0.2, 2)

#cProfile.run("main()")     
#if __name__ == "__main__": 
#  main(14, 1 , 0.2, 2)
  
if __name__ == '__main__':
    main(sys.argv[1:])  

#plt.plot(flow, energy)