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

import J_Dev5_cml2

import numpy as np

import matplotlib.pyplot as plt # Plotting stuff

import argparse #for cmd line args

import datetime#for file names

import csv



### Import Magic numbers to determine number of sp states for given Nmax ###

f = open("Magic_Numbers.txt", "r")

Magic_Numbers = f.read() #Magic numbers from N_max = 0 : N_max = 200
#Magic_Numbers[0] is 0th shell  

# convert them into list of ints using pre_process
Magic_Numbers = J_Dev5_cml2.pre_process(Magic_Numbers)





######## Command line args #########

parser = argparse.ArgumentParser(description="""Input arguments: --A=... 
                                 --Max_Nmax=... --den=... --g=... """)

parser.add_argument("--A", required=True, help="""Number of particles in box.
                    Only number of holes in closed shells are allowed""",
                    choices = Magic_Numbers, type=int)


parser.add_argument("--Max_Nmax", required=True, type=int,
                    help="""This specifies the maximum number of shells 
                    above the closed shells occupied by the particles""")


parser.add_argument("--Min_Nmax", required=False, type=int,
                    help="""This specifies the minimum number of shells 
                    above the closed shells occupied by the particles""",
                    default = 1)

#parser.add_argument("--den", required=True, type=float, 
#                    help="Density of particles in box")
  


parser.add_argument("--g", required=False, type=int, choices=[2,4],
                    default= 2, help="=4 for SNM, & =2 for PNM")



parser.add_argument("--DE", required=False, type=int, default=0,
                    choices=[0,1], help="0=MBPT(3) off and 1=MBPT(3) on")


parser.add_argument("--T", required=False, type=int, choices=range(1,17),
                    default=6, help="BCH expansion will go up to T terms") 


parser.add_argument("--step", required=False, type=float, default = 1.0,
                    help="Step size in s: 1.0 seems to work fine")  

parser.add_argument("--num_pts", required=True, type=int,
                    help="This specifies the number of densities in EOS") 

args = parser.parse_args()



A = args.A #specify number of particles you want to include

Max_N_Max = args.Max_Nmax#How many shells above the holes you want to include.

Min_N_Max = args.Min_Nmax#How many shells above the holes you want to include.

#Unlike before, N_max cannot equal 0. We need particle states!

#rho = args.den #Define density in fm^-3  

#Additional inputs below if I want to change those parameters from EOS script

degen = args.g #If degen=4, then the matter is symmetric and isospin 
#projections are considered. If degen=2, there is only one type of particle, 
#so isospin projections are not considered. Degen is only either 2 or 4.

DE_flag = args.DE#Current 3rd order implementation  rightnow is 
#realllly slow and should not be used, but here it is

trunc = args.T #Set max # of terms in BCH and Magnus expansion

ds = args.step #obtain step size if requested

num_points = args.num_pts# number of densities used in EOS



######## Initializations ############

E = []

fraction_zero = ["frac_zero"]

diff_zero = ["frac_non-zero"]

num_sp_states = ["num_sp_states"]

valid_N_maxs = ["N_max"]

#What Nmax corresponds to number of holes states?
Hole_Nmax = Magic_Numbers.index(A)

ranger = range(Min_N_Max, Max_N_Max + 1)

#how much is the norm of Omega changing as I increase Nmax?
Omega_norm = ["Omega_norm"]#norm should be normalized to number of matrix elem

#fractions = []

rhos = np.linspace(0.05, 0.2, num_points)

Collection_of_Outputs = [["den", *list(rhos)]]#first column is range of densities

Residual_Collection_of_Outputs = [] #first column is Nmax range

Temp_Last = []#so I can stick norm(Omega(s->infty))(rho,Nmax) towards 
#the end of Collection_of_Outputs

Temp_Avg = []#so I can stick mean(norm(Omega(s)))(rho,Nmax) at end of 
#Collection_of_Outputs

Temp_Last_ham = []#so I can stick norm(Gamma(s->infty))(rho,Nmax) towards 
#the end of Collection_of_Outputs

Temp_Avg_ham = []#so I can stick mean(norm(Gamma(s)))(rho,Nmax) at end of 
#Collection_of_Outputs

####### Perform EOS for each Nmax within range requested ##########

for Nmax in ranger:
  
  if(Magic_Numbers[Hole_Nmax + Nmax] != Magic_Numbers[Hole_Nmax + Nmax - 1]): 
  #only do IMSRG once if Nmax = 5, and 6 (for example) have same number of 
  #states
    
    valid_N_maxs.append(Nmax)#store this N_max as valid
    
    E_for_dis_Nmax = ["E (N_max = " + str(Nmax) + ")"]
    
    Last_norm_over_rhos = ["Fin_Omega_norm (N_max = " + str(Nmax) + ")"]
    
    Avg_norm_over_rhos = ["Avg_Omega_norm (N_max = " + str(Nmax) + ")"]
    
    GLast_norm_over_rhos = ["Fin_Gamma_norm (N_max = " + str(Nmax) + ")"]
    
    GAvg_norm_over_rhos = ["Avg_Gamma_norm (N_max = " + str(Nmax) + ")"]
     
    for rho in rhos:
    
      J_Dev5_cml2.main(["--A", str(A), "--Nmax", str(Nmax), "--den", str(rho),
                   "--g", str(degen), "--DE", str(DE_flag), "--T", str(trunc),
                   "--step", str(ds)])#execute main
      
      res = J_Dev5_cml2.results[0] #retrieve results from completed run
      
      E_for_dis_Nmax.append(res[0]/A)#get energies
      
      Last_norm_over_rhos.append(res[7])#get norm as function of density
      
      Avg_norm_over_rhos.append(res[8]) #get avg scaled norm as function of density
      
      GLast_norm_over_rhos.append(res[9])#get norm as function of density
      
      GAvg_norm_over_rhos.append(res[10]) #get avg scaled norm as function of density
      
      
    fraction_zero.append(res[4])# obtain fraction of zero-blks in omega
    
    diff_zero.append(res[5])# obtain number of non-zero blks
    
    num_sp_states.append(res[6]) # get number of s.p states
    
    Omega_norm.append(res[7])# get norm normalized to number of sp states
         
    Collection_of_Outputs.append(E_for_dis_Nmax)
    
    Temp_Last.append(Last_norm_over_rhos)
    
    Temp_Avg.append(Avg_norm_over_rhos)

    Temp_Last_ham.append(GLast_norm_over_rhos)
    
    Temp_Avg_ham.append(GAvg_norm_over_rhos)
    
  else:

    print("N_max = ", Nmax, " has the same number of sp states as ",
          "N_max-1 = ", Nmax-1, " so let's skip it! " )   
    
#Number of columns in Collection_of_Outputs = 3*len(valid_N_maxs) + 1
#+1 is for first column with densities    
#print(Collection_of_Outputs )    
Collection_of_Outputs = (Collection_of_Outputs + Temp_Last + Temp_Avg + 
                         Temp_Last_ham + Temp_Avg_ham)

#print(Collection_of_Outputs )  
    
#Append for output file titled "Res_A=..."

Residual_Collection_of_Outputs.append(valid_N_maxs) 
Residual_Collection_of_Outputs.append(num_sp_states)    
Residual_Collection_of_Outputs.append(fraction_zero)    
Residual_Collection_of_Outputs.append(diff_zero)
Residual_Collection_of_Outputs.append(Omega_norm)     
 

#print(Residual_Collection_of_Outputs)

#print(list(zip(*Residual_Collection_of_Outputs)))
######## Write to files outside ############

#tiempo = datetime.datetime.now()#Both files will have same time stamp
        
#title = ('A='+ str(A) + "_" + "(N_Max =" + str(Min_N_Max)+":"+str(Max_N_Max) 
#  + ")"+ "@"+ "("  + str(tiempo) +")"+".csv") 
 
title = ('A='+ str(A) + '_' + '(N_max=' + str(Min_N_Max)+':'+str(Max_N_Max) 
  + ')'+ '_num_pts=' + str(num_points) +'.csv')  

with open(title, 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(zip(*Collection_of_Outputs))#* is to unpack list so 
  #Collection_of_Outputs goes from list of lists to just a lists that can 
  
  

#title = ('Res_A='+ str(A) + "_" + "(N_Max =" + str(Min_N_Max) 
#  +":"+str(Max_N_Max) + ")"+ "@"+ "("  + str(tiempo) +")"+".csv") 
 
title = ('Res_A='+ str(A) + '_' + '(N_max=' + str(Min_N_Max)+':'+str(Max_N_Max) 
  + ')'+ '_num_pts=' + str(num_points) +'.csv')

with open(title, 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(zip(*Residual_Collection_of_Outputs)) 

