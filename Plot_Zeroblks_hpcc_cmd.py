#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:59:09 2020

@author: YaniUdiani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:30:50 2020

@author: YaniUdiani
"""

import J_Dev5_cml

import numpy as np

import matplotlib.pyplot as plt # Plotting stuff

import argparse #for cmd line args

import datetime#for file names



def test_similarities(mom_for_zero_blks):
  
      
  how_many_in_Nmax_plus_one = 0
  
  for elements in mom_for_zero_blks[0]:
    
    if(elements in mom_for_zero_blks[1]):
      
      how_many_in_Nmax_plus_one += 1
      
      
  return how_many_in_Nmax_plus_one/len(mom_for_zero_blks[0])



f = open("Magic_Numbers.txt", "r")

Magic_Numbers = f.read() #Magic numbers from N_max = 0 : N_max = 200
#Magic_Numbers[0] is 0th shell  

Magic_Numbers = J_Dev5_cml.pre_process(Magic_Numbers)# convert them into list of ints

#Magic.append(Magic_Numbers)


parser = argparse.ArgumentParser(description='Input arguments: --A=... --Max_Nmax=... --den=... --g=... ')

parser.add_argument("--A", required=True, type=int, 
help="Number of particles in box. Only number of holes in closed shells are allowed",
choices = Magic_Numbers)


parser.add_argument("--Max_Nmax", required=True, type=int,
                    help="This specifies the model space")



parser.add_argument("--den", required=True, type=float, 
                    help="Density of particles in box")
  


parser.add_argument("--g", required=False, type=int, choices=[2,4],
                    default= 2, help="=4 for SNM, & =2 for PNM")



parser.add_argument("--DE", required=False, type=int, default=0,
                    choices=[0,1], help="0=MBPT(3) off and 1=MBPT(3) on")


parser.add_argument("--T", required=False, type=int, choices=range(1,17), default=6, 
                    help="BCH expansion will go up to T terms") 


parser.add_argument("--step", required=False, type=float, default = 1.0,
                    help="Step size in s: 1.0 seems to work fine")  
 

args = parser.parse_args()



A = args.A #specify number of particles you want to include

Max_N_Max = args.Max_Nmax#How many shells above the holes you want to include. 

#Unlike before, N_max cannot equal 0. We need particle states!

rho = args.den #Define density in fm^-3  

#Additional inputs below if I want to change those parameters from EOS script

degen = args.g #If degen=4, then the matter is symmetric and isospin 
#projections are considered. If degen=2, there is only one type of particle, 
#so isospin projections are not considered. Degen is only either 2 or 4.

DE_flag = args.DE#Current 3rd order implementation  rightnow is 
#realllly slow and should not be used, but here it is

trunc = args.T #Set max # of terms in BCH and Magnus expansion

ds = args.step #obtain step size if requested



E = []

fraction_zero = []

diff_zero = []

num_sp_states = []


Hole_Nmax = Magic_Numbers.index(A)

ranger = range(1, Max_N_Max + 1)

mom_for_zero_blks = []#keep track of momenta for blocks of omega that are 
#zero as N_max is changing. The question I have is that, for a  given N_max
#the zero blocks of Omega are mom_for_zero_blks[N_max], now N_max -> N_max + 1.
#Are the blocks mom_for_zero_blks[N_max] present in mom_for_zero_blks[N_max+1]?

Omega_norm = [] #how much is the norm of Omega changing as I increase Nmax?

fractions = []

for Nmax in ranger:
  
  if(Magic_Numbers[Hole_Nmax + Nmax] != Magic_Numbers[Hole_Nmax + Nmax - 1]): 
  #only do IMSRG once if Nmax = 5, and 6 (for example) have same number of states
  
    J_Dev5_cml.main(["--A", str(A), "--Nmax", str(Nmax), "--den", str(rho)])#execute main
    
    res = J_Dev5_cml.results[0] #retrieve results from completed run
    
    E.append(res[0]/A)#get energies

    fraction_zero.append(res[4])# obtain fraction of zero-blks in omega
    
    diff_zero.append(res[5])# obtain number of non-zero blks
    
    num_sp_states.append(res[6]) # get number of s.p states
    
    Zero_Blocks = res[7]
    
    blk_nums_to_mom = res[8]
        
    momenta = []
    
    for block in Zero_Blocks:
      
      momenta.append( blk_nums_to_mom[block] )
    
    mom_for_zero_blks.append(momenta)
   
    
    if len(mom_for_zero_blks) == 2:
  
      
      fractions.append(test_similarities(mom_for_zero_blks))
      
      mom_for_zero_blks.clear()#to save on memory
      
    Omega_norm.append (J_Dev5_cml.calc_full2B_norm(res[3], range(len(res[3])) ) )
    
    
      
        
  
print("fraction of momenta for some N_max in momenta of zero blks for N_max + 1 : ", fractions) 

#test_similarities(mom_for_zero_blks)

plt.plot(num_sp_states, E)
plt.ylabel("E/A (MeV)")
plt.xlabel("Number of s.p. states")
#plt.tick_params(width=10)
plt.title("E/A for N_max = " + str(ranger[0]) + " : " + str(ranger[Max_N_Max-1]))
plt.tight_layout()
plt.savefig("Convergence_(" + str (A) + "_" +str(rho) +  "_" +str(Max_N_Max) +")_" 
            + str(datetime.datetime.now()) +".png" ,format = "png", dpi= 500)

plt.figure()
plt.plot(num_sp_states, fraction_zero)
plt.ylabel("Fraction Zero Blks ")
plt.xlabel("Number of s.p. states")
plt.title(" Fraction of zero blocks in Omega for N_max = " + str(ranger[0]) + " : " + str(ranger[Max_N_Max-1]))
plt.savefig("ZeroOmegabasis_(" + str (A) + "_" +str(rho) +  "_" +str(Max_N_Max) +")_" 
            + str(datetime.datetime.now()) +".png" ,format = "png", dpi= 500)

#plt.figure()
#plt.plot(num_sp_states, diff_zero)
#plt.ylabel("Number of Non-Zero Blks ")
#plt.xlabel("Number of s.p. states")
#plt.title(" Number of Non-Zero Blks in Omega for N_max = " + str(ranger[0]) + " : " + str(ranger[Max_N_Max-1]))

plt.figure()
plt.plot(num_sp_states, Omega_norm)
plt.ylabel("Norm of Omega ")
plt.xlabel("Number of s.p. states")
plt.title(" Norm of Omega for N_max = " + str(ranger[0]) + " : " + str(ranger[Max_N_Max-1]))
plt.savefig("Omega_Norm_(" + str (A) + "_" +str(rho) +  "_" +str(Max_N_Max) +")_" 
            + str(datetime.datetime.now()) +".png" ,format = "png", dpi= 500)