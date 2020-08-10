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

import pandas as pd

f = open("Magic_Numbers.txt", "r")

Magic_Numbers = f.read() #Magic numbers from N_max = 0 : N_max = 200
#Magic_Numbers[0] is 0th shell  

Magic_Numbers = J_Dev5_cml.pre_process(Magic_Numbers)# convert them into list of ints

#Magic.append(Magic_Numbers)


parser = argparse.ArgumentParser(description='Input arguments: --A=... --Nmax=... --num_pts=... --g=... ')


parser.add_argument("--A", required=True, type=int, 
help="Number of particles in box. Only number of holes in closed shells are allowed",
choices = Magic_Numbers)


parser.add_argument("--Nmax", required=True, type=int,
                    help="This specifies the model space")



parser.add_argument("--num_pts", required=True, type=int,
                    help="This specifies the number of densities in EOS")


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

N_Max = args.Nmax#How many shells above the holes you want to include. 

#Unlike before, N_max cannot equal 0. We need particle states!


num_points = args.num_pts# number of densities used in EOS


#Additional inputs below if I want to change those parameters from EOS script

degen = args.g #If degen=4, then the matter is symmetric and isospin 
#projections are considered. If degen=2, there is only one type of particle, 
#so isospin projections are not considered. Degen is only either 2 or 4.

DE_flag = args.DE#Current 3rd order implementation  rightnow is 
#realllly slow and should not be used, but here it is

trunc = args.T #Set max # of terms in BCH and Magnus expansion

ds = args.step #obtain step size if requested


#rhos = np.linspace(0.05, 0.2, num_points)

#rhos = np.array([0.02, 0.04, 0.06, 0.08, 0.1])

rhos = np.array([0.12, 0.14, 0.16, 0.18, 0.2])

num_points = len(rhos)

E = []

fraction_zero = []

for rho in rhos:
  
  J_Dev5_cml.main(["--A", str(A), "--Nmax", str(N_Max), "--den", str(rho),
                   "--g", str(degen), "--DE", str(DE_flag), "--T", str(trunc),
                   "--step", str(ds) ])#execute main
  
  res = J_Dev5_cml.results[0] #retrieve results from completed run
  
  E.append(res[0]/A)
  
  fraction_zero.append(res[4])
  

num_sp_states = res[6] # get number of s.p states  


plt.plot(rhos, E)
plt.ylabel("E/A (MeV)")
plt.xlabel("density (fm^-3)")
plt.title("EOS for " + str (A) + " particles with " + str(num_sp_states) + " sp states")
#plt.savefig("EOS_(" + str (A) + "_" +str(N_Max) + "_" +str(rho) 
#+ " (" + str(datetime.datetime.now()) + ")" )
plt.tight_layout()
plt.savefig("EOS_(" + str (A) + "_" +str(N_Max) + "_" + str(num_points) + ")_" 
            + str(datetime.datetime.now()) +".png" ,format = "png", dpi= 500)

eos_data = pd.DataFrame({"den": rhos, "E": E})

np.savetxt(r'eos' + "_" + str (A) + "_" +str(N_Max) + "_" + str(num_points) + "_"
            + str(datetime.datetime.now()) + '.txt', eos_data.values, fmt='%1.9f')

#with open('EOS.txt', 'w') as f:
#    for item in E:
#        f.write("%s\n" % item)
#
#with open('EOS.txt', 'w') as f:
#    for item in rhos:
#        f.write("%s\n" % item)
        
plt.figure()
plt.plot(rhos, fraction_zero)
plt.ylabel("Fraction Zero Blks ")
plt.xlabel("density (fm^-3)")
plt.title(" Fraction of zero blocks in Omega (A = " + str (A) + ", num sp states = " + str(num_sp_states) +")")
plt.tight_layout()
plt.savefig("ZeroOmega_(" + str (A) + "_" +str(N_Max) + "_" + str(num_points) + ")_" 
            + str(datetime.datetime.now())+ ".png",format = "png", dpi= 500)

eos_data_1 = pd.DataFrame({"den": rhos, "fraction_zero": fraction_zero})

np.savetxt(r'zeroblks' + "_" + str (A) + "_" +str(N_Max) + "_" + str(num_points) + "_"
            + str(datetime.datetime.now()) +'.txt', eos_data_1.values, fmt='%1.9f')
#plt.tick_params(width=10)
