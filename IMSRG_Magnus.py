#!/usr/bin/env python

#------------------------------------------------------------------------------
# IMSRG_Magnus.py
#
# author:   Yani Udiani
# Adapted from Heiko's IMSRG
# date:     Nov, 1, 2018
# 
# tested with Python 3
# 
# Solves the pairing model for four particles in a basis of four doubly 
# degenerate states by means of an In-Medium Similarity Renormalization using the Magnus expansion
# Group (IMSRG) flow.
#
#------------------------------------------------------------------------------

import numpy as np
from numpy import array, dot, diag, reshape, transpose
from scipy.linalg import eigvalsh
from scipy.integrate import odeint, ode
from numpy import linalg as LA # For Frobenius norm
from fractions import Fraction as Fr # For Bernoulli Numbers
from sys import argv
import matplotlib.pyplot as plt # Plotting stuff

#-----------------------------------------------------------------------------------
# basis and index functions
#-----------------------------------------------------------------------------------
flow=[]
energy=[]
def construct_basis_2B(holes, particles):
  basis = []
  for i in holes:
    for j in holes:
      basis.append((i, j))

  for i in holes:
    for a in particles:
      basis.append((i, a))

  for a in particles:
    for i in holes:
      basis.append((a, i))

  for a in particles:
    for b in particles:
      basis.append((a, b))

  return basis


def construct_basis_ph2B(holes, particles):
  basis = []
  for i in holes:
    for j in holes:
      basis.append((i, j))

  for i in holes:
    for a in particles:
      basis.append((i, a))

  for a in particles:
    for i in holes:
      basis.append((a, i))

  for a in particles:
    for b in particles:
      basis.append((a, b))

  return basis


#
# We use dictionaries for the reverse lookup of state indices
#
def construct_index_2B(bas2B):
  index = { }
  for i, state in enumerate(bas2B):
    index[state] = i

  return index



#-----------------------------------------------------------------------------------
# transform matrices to particle-hole representation
#-----------------------------------------------------------------------------------
def ph_transform_2B(Gamma, bas2B, idx2B, basph2B, idxph2B):
  dim = len(basph2B)
  Gamma_ph = np.zeros((dim, dim))

  for i1, (a,b) in enumerate(basph2B):
    for i2, (c, d) in enumerate(basph2B):
      Gamma_ph[i1, i2] -= Gamma[idx2B[(a,d)], idx2B[(c,b)]]

  return Gamma_ph

def inverse_ph_transform_2B(Gamma_ph, bas2B, idx2B, basph2B, idxph2B):
  dim = len(bas2B)
  Gamma = np.zeros((dim, dim))

  for i1, (a,b) in enumerate(bas2B):
    for i2, (c, d) in enumerate(bas2B):
      Gamma[i1, i2] -= Gamma_ph[idxph2B[(a,d)], idxph2B[(c,b)]]
  
  return Gamma

#-----------------------------------------------------------------------------------
# commutator of matrices
#-----------------------------------------------------------------------------------
def commutator(a,b):
  return dot(a,b) - dot(b,a)

#-----------------------------------------------------------------------------------
# norms of off-diagonal Hamiltonian pieces
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
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  norm = 0.0
  for a in particles:    
    for b in particles:
      for i in holes:
        for j in holes:
          norm += Gamma[idx2B[(a,b)],idx2B[(i,j)]]**2 + Gamma[idx2B[(i,j)],idx2B[(a,b)]]**2

  return np.sqrt(norm)

#-----------------------------------------------------------------------------------
# occupation number matrices
#-----------------------------------------------------------------------------------
def construct_occupation_1B(bas1B, holes, particles):
  dim = len(bas1B)
  occ = np.zeros(dim)

  for i in holes:
    occ[i] = 1.

  return occ

# diagonal matrix: n_a - n_b
def construct_occupationA_2B(bas2B, occ1B):
  dim = len(bas2B)
  occ = np.zeros((dim,dim))

  for i1, (i,j) in enumerate(bas2B):
    occ[i1, i1] = occ1B[i] - occ1B[j]

  return occ


# diagonal matrix: 1 - n_a - n_b
def construct_occupationB_2B(bas2B, occ1B):
  dim = len(bas2B)
  occ = np.zeros((dim,dim))

  for i1, (i,j) in enumerate(bas2B):
    occ[i1, i1] = 1. - occ1B[i] - occ1B[j]

  return occ

# diagonal matrix: n_a * n_b
def construct_occupationC_2B(bas2B, occ1B):
  dim = len(bas2B)
  occ = np.zeros((dim,dim))

  for i1, (i,j) in enumerate(bas2B):
    occ[i1, i1] = occ1B[i] * occ1B[j]

  return occ

#-----------------------------------------------------------------------------------
# generators
#-----------------------------------------------------------------------------------
def Bernoulli_generator(y): #returns list of Bernoulli numbers indexed by 0 to (y-1)
    def bernoulli2(): # Function taken online to calculate Bernouli sequence
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

def eta_brillouin(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      # (1-n_a)n_i - n_a(1-n_i) = n_i - n_a
      eta1B[a, i] =  f[a,i]
      eta1B[i, a] = -f[a,i]

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]]

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B

def eta_imtime(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      dE = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = np.sign(dE)*f[a,i]
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          dE = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = np.sign(dE)*Gamma[idx2B[(a,b)], idx2B[(i,j)]]

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_white(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_white_mp(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
          )

          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B

def eta_white_atan(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = 0.5 * np.arctan(2 * f[a,i]/denom)
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j] 
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]] 
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = 0.5 * np.arctan(2 * Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom)

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_wegner(f, Gamma, user_data):

  dim1B     = user_data["dim1B"]
  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas2B     = user_data["bas2B"]
  basph2B   = user_data["basph2B"]
  idx2B     = user_data["idx2B"]
  idxph2B   = user_data["idxph2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]


  # split Hamiltonian in diagonal and off-diagonal parts
  fd      = np.zeros_like(f)
  fod     = np.zeros_like(f)
  Gammad  = np.zeros_like(Gamma)
  Gammaod = np.zeros_like(Gamma)

  for a in particles:
    for i in holes:
      fod[a, i] = f[a,i]
      fod[i, a] = f[i,a]
  fd = f - fod

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          Gammaod[idx2B[(a,b)], idx2B[(i,j)]] = Gamma[idx2B[(a,b)], idx2B[(i,j)]]
          Gammaod[idx2B[(i,j)], idx2B[(a,b)]] = Gamma[idx2B[(i,j)], idx2B[(a,b)]]
  Gammad = Gamma - Gammaod


  #############################        
  # one-body flow equation  
  eta1B  = np.zeros_like(f)

  # 1B - 1B
  eta1B += commutator(fd, fod)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        for a in particles:
          eta1B[p,q] += (
            fd[i,a]  * Gammaod[idx2B[(a, p)], idx2B[(i, q)]] 
            - fd[a,i]  * Gammaod[idx2B[(i, p)], idx2B[(a, q)]] 
            - fod[i,a] * Gammad[idx2B[(a, p)], idx2B[(i, q)]] 
            + fod[a,i] * Gammad[idx2B[(i, p)], idx2B[(a, q)]]
          )

  # 2B - 2B
  # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
  GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        eta1B[p,q] += (
          0.5*GammaGamma[idx2B[(i,p)], idx2B[(i,q)]] 
          - transpose(GammaGamma)[idx2B[(i,p)], idx2B[(i,q)]]
        )

  GammaGamma = dot(Gammad, dot(occC_2B, Gammaod))
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        eta1B[p,q] += (
          0.5*GammaGamma[idx2B[(r,p)], idx2B[(r,q)]] 
          + transpose(GammaGamma)[idx2B[(r,p)], idx2B[(r,q)]] 
        )


  #############################        
  # two-body flow equation  
  eta2B = np.zeros_like(Gamma)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          for t in range(dim1B):
            eta2B[idx2B[(p,q)],idx2B[(r,s)]] += (
              fd[p,t] * Gammaod[idx2B[(t,q)],idx2B[(r,s)]] 
              + fd[q,t] * Gammaod[idx2B[(p,t)],idx2B[(r,s)]] 
              - fd[t,r] * Gammaod[idx2B[(p,q)],idx2B[(t,s)]] 
              - fd[t,s] * Gammaod[idx2B[(p,q)],idx2B[(r,t)]]
              - fod[p,t] * Gammad[idx2B[(t,q)],idx2B[(r,s)]] 
              - fod[q,t] * Gammad[idx2B[(p,t)],idx2B[(r,s)]] 
              + fod[t,r] * Gammad[idx2B[(p,q)],idx2B[(t,s)]] 
              + fod[t,s] * Gammad[idx2B[(p,q)],idx2B[(r,t)]]
            )

  
  # 2B - 2B - particle and hole ladders
  # Gammad.occB.Gammaod
  GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))

  eta2B += 0.5 * (GammaGamma - transpose(GammaGamma))

  # 2B - 2B - particle-hole chain
  
  # transform matrices to particle-hole representation and calculate 
  # Gammad_ph.occA_ph.Gammaod_ph
  Gammad_ph = ph_transform_2B(Gammad, bas2B, idx2B, basph2B, idxph2B)
  Gammaod_ph = ph_transform_2B(Gammaod, bas2B, idx2B, basph2B, idxph2B)

  GammaGamma_ph = dot(Gammad_ph, dot(occphA_2B, Gammaod_ph))

  # transform back to standard representation
  GammaGamma    = inverse_ph_transform_2B(GammaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

  # commutator / antisymmetrization
  work = np.zeros_like(GammaGamma)
  for i1, (i,j) in enumerate(bas2B):
    for i2, (k,l) in enumerate(bas2B):
      work[i1, i2] -= (
        GammaGamma[i1, i2] 
        - GammaGamma[idx2B[(j,i)], i2] 
        - GammaGamma[i1, idx2B[(l,k)]] 
        + GammaGamma[idx2B[(j,i)], idx2B[(l,k)]]
      )
  GammaGamma = work

  eta2B += GammaGamma


  return eta1B, eta2B


#-----------------------------------------------------------------------------------
# derivatives : My changes start here
#-----------------------------------------------------------------------------------
  
def special_commutator(x,y,user_data): # takes in either 1 or 2 body matrices to output 0,1, and 2 body commutators
    dim1B     = user_data["dim1B"]
    holes     = user_data["holes"]
    particles = user_data["particles"]  
    idx2B     = user_data["idx2B"]
    states=particles+holes #combine particles and holes for easy indexing
    #Initializations 
    Output_1B=np.zeros((dim1B,dim1B))
    Output_2B=np.zeros((dim1B**2,dim1B**2))
    Output_0B=0.0
    # The remaining lines of the function brute force the commutators. I will convert some of these to matrix multiplication once I fix my bugs :)
    if(np.size(x)== len(states)**2 and np.size(y)== len(states)**2):#1B-1B
        Output_1B+=commutator(x,y)#regular commutator
        for i in holes:# 0B Correction
            for j in particles:
                Output_0B+=x[i,j]*y[j,i]-x[j,i]*y[i,j]
                
    if(np.size(x)== len(states)**4 and np.size(y)== len(states)**2):#2B-1B 
        for i in states:
            for j in states:
                for a in holes:
                    for b in particles:
                        #Output_1B[i,j]+=-(int(a in holes)-int(b in holes))*y[i,j]*x[idx2B[(b,i)],idx2B[(a,j)]]
                        Output_1B[i,j]+=-(y[i,j]*x[idx2B[(b,i)],idx2B[(a,j)]]-y[i,j]*x[idx2B[(a,i)],idx2B[(b,j)]])
                        
        for i in states:
            for j in states:
                for k in states:
                    for l in states:
                        for a in states:
                            Output_2B[idx2B[(i,j)],idx2B[(k,l)]]+=-( (y[i,a]*x[idx2B[(a,j)],idx2B[(k,l)]]-y[j,a]*x[idx2B[(a,i)],idx2B[(k,l)]]) - (y[a,k]*x[idx2B[(i,j)],idx2B[(a,l)]]-y[a,l]*x[idx2B[(i,j)],idx2B[(a,k)]]))
                            
    if(np.size(x)== len(states)**2 and np.size(y)== len(states)**4):#1B-2B 
        for i in states:
            for j in states:
                for a in holes:
                    for b in particles:
                        #Output_1B[i,j]+=(int(a in holes)-int(b in holes))*x[i,j]*y[idx2B[(b,i)],idx2B[(a,j)]]
                        Output_1B[i,j]+=-(x[i,j]*y[idx2B[(b,i)],idx2B[(a,j)]]-x[i,j]*y[idx2B[(a,i)],idx2B[(b,j)]])
                        
        for i in states:
            for j in states:
                for k in states:
                    for l in states:
                        for a in states:
                            Output_2B[idx2B[(i,j)],idx2B[(k,l)]]+= (x[i,a]*y[idx2B[(a,j)],idx2B[(k,l)]]-x[j,a]*y[idx2B[(a,i)],idx2B[(k,l)]]) - (x[a,k]*y[idx2B[(i,j)],idx2B[(a,l)]]-x[a,l]*y[idx2B[(i,j)],idx2B[(a,k)]]) 

    if(np.size(x)== len(states)**4 and np.size(y)== len(states)**4):# 2B-2B
        for i in holes:
            for j in holes:
                for k in particles:
                    for l in particles:
                        #Output_0B+=0.25*(int(i in holes)*int(j in holes)*(1-int(k in holes))*(1-int(l in holes)))*(x[idx2B[(i,j)],idx2B[(k,l)]]*y[idx2B[(k,l)],idx2B[(i,j)]]-x[idx2B[(k,l)],idx2B[(i,j)]]*y[idx2B[(i,j)],idx2B[(k,l)]])
                        Output_0B+=0.25*(x[idx2B[(i,j)],idx2B[(k,l)]]*y[idx2B[(k,l)],idx2B[(i,j)]]-x[idx2B[(k,l)],idx2B[(i,j)]]*y[idx2B[(i,j)],idx2B[(k,l)]])                       
        for i in states:
            for j in states:
                for a in states:
                    for b in states:
                        for c in states:
                            Output_1B[i,j]+=0.5*( x[idx2B[(c,i)],idx2B[(a,b)]]*y[idx2B[(a,b)],idx2B[(c,j)]] - x[idx2B[(a,b)],idx2B[(c,j)]]*y[idx2B[(c,i)],idx2B[(a,b)]] )*((1-int(a in holes))*(1-int(b in holes))*int(c in holes)+(1-int(c in holes))*int(a in holes)*int(b in holes))
                            
        for i in states:
            for j in states:
                for k in states:
                    for l in states:
                        for a in states:
                            for b in states:
                                Output_2B[idx2B[(i,j)],idx2B[(k,l)]]+=(0.5*(x[idx2B[(i,j)],idx2B[(a,b)]]*y[idx2B[(a,b)],idx2B[(k,l)]]-x[idx2B[(a,b)],idx2B[(k,l)]]*y[idx2B[(i,j)],idx2B[(a,b)]])*(1-int(a in holes)-int(b in holes))+(int(a in holes)-int(b in holes))*(x[idx2B[(a,i)],idx2B[(b,k)]]*y[idx2B[(b,j)],idx2B[(a,l)]]-x[idx2B[(a,j)],idx2B[(b,k)]]*y[idx2B[(b,i)],idx2B[(a,l)]]-x[idx2B[(a,i)],idx2B[(b,l)]]*y[idx2B[(b,j)],idx2B[(a,k)]]+x[idx2B[(a,j)],idx2B[(b,l)]]*y[idx2B[(b,i)],idx2B[(a,k)]]))
    
    return Output_0B, Output_1B, Output_2B   

def RHS_Cal(Omega_1B,Omega_2B,Eta_1B,Eta_2B, user_data):# Generates right hand side of flow equation to be fed to diffy solver to obtain new Omega
    bn2 = user_data["bn2"]#import Bernoulli numbers
    #Initializations
    RHS_0B=0. 
    RHS_1B=np.zeros_like(Omega_1B)
    RHS_2B=np.zeros_like(Omega_2B)
    special_commutator=user_data["special_commutator"]
    for n in range(len(bn2)):
        if n==0: 
            nth_2B=Eta_2B #Initial commutators defined to be Eta
            nth_1B=Eta_1B
            nth_0B=0.0
        else:
            # The next lines aggregate the current 0B, 1B, and 2B nth commutators from the Magnus expansion
            nth_0B=special_commutator(Omega_1B,nth_1B, user_data)[0]
            nth_0B=nth_0B+special_commutator(Omega_2B,nth_2B,user_data)[0]
            
            nth_1B=special_commutator(Omega_1B,nth_1B, user_data)[1]
            nth_1B+=special_commutator(Omega_1B,nth_2B,user_data)[1]
            nth_1B+=special_commutator(Omega_2B,nth_1B, user_data)[1]
            nth_1B+=special_commutator(Omega_2B,nth_2B, user_data)[1]
            
            nth_2B=special_commutator(Omega_1B,nth_2B, user_data)[2]
            nth_2B+=special_commutator(Omega_2B,nth_1B, user_data)[2]
            nth_2B+=special_commutator(Omega_2B,nth_2B, user_data)[2]
            
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        shiftymatrix_0B=nth_0B/np.math.factorial(n)
        shiftymatrix_1B=nth_1B/np.math.factorial(n)
        shiftymatrix_2B=nth_2B/np.math.factorial(n)
        if ((LA.norm(shiftymatrix_1B)+ LA.norm(shiftymatrix_2B))<1e-10):
            break        
        RHS_0B+=bn2[n]*shiftymatrix_0B
        RHS_1B+=bn2[n]*shiftymatrix_1B
        RHS_2B+=bn2[n]*shiftymatrix_2B
        
    return RHS_0B, RHS_1B, RHS_2B

def Transformed_Ham(Omega_1B, Omega_2B, user_data):# Generates new Hamiltonian by performing BCH expansion
    bn2 = user_data["bn2"]#import Bernoulli numbers
    E=user_data["E"]#Import initial energy
    f=user_data["f"]#import initial one body hamiltonian
    Gamma=user_data["Gamma"]#import initial two body hamiltonian
    #Initializations
    H_0B=0.
    H_1B=np.zeros_like(f)
    H_2B=np.zeros_like(Gamma)
    special_commutator=user_data["special_commutator"] # I really don't need this. Come back to clean it up later
    for n in range(len(bn2)):
        if n==0: 
            nth_2B=Gamma #Initial commutators defined to be initial Hamiltonian
            nth_1B=f
            nth_0B=E
        else:
            # The next lines aggregate the current 0B, 1B, and 2B nth commutators from the Magnus expansion
            nth_0B=special_commutator(Omega_1B,nth_1B, user_data)[0]
            nth_0B+=special_commutator(Omega_2B,nth_2B,user_data)[0]
            
            nth_1B=special_commutator(Omega_1B,nth_1B, user_data)[1]
            nth_1B+=special_commutator(Omega_1B,nth_2B,user_data)[1]
            nth_1B+=special_commutator(Omega_2B,nth_1B, user_data)[1]
            nth_1B+=special_commutator(Omega_2B,nth_2B, user_data)[1]
            
            nth_2B=special_commutator(Omega_1B,nth_2B, user_data)[2]
            nth_2B+=special_commutator(Omega_2B,nth_1B, user_data)[2]
            nth_2B+=special_commutator(Omega_2B,nth_2B, user_data)[2]
            
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        shiftymatrix_0B=nth_0B/np.math.factorial(n)
        shiftymatrix_1B=nth_1B/np.math.factorial(n)
        shiftymatrix_2B=nth_2B/np.math.factorial(n)
        if ((LA.norm(shiftymatrix_1B)+LA.norm(shiftymatrix_2B))<1e-10):
            break
        H_0B+=shiftymatrix_0B
        H_1B+=shiftymatrix_1B
        H_2B+=shiftymatrix_2B
    return H_0B, H_1B, H_2B
#-----------------------------------------------------------------------------------
# derivative wrapper
#-----------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, dim2B):
  
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
  one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))

  ptr += dim1B*dim1B
  two_body = reshape(y[ptr:ptr+dim2B*dim2B], (dim2B, dim2B))

  return zero_body,one_body,two_body


def derivative_wrapper(t, y, user_data):

  dim1B = user_data["dim1B"]
  dim2B = dim1B*dim1B
  calc_eta  = user_data["calc_eta"]# Function to calculate generator
  calc_rhs  = user_data["calc_rhs"]    #function to calculate RHS
  
  # extract operator pieces from solution vector
  Omega_0B,Omega_1B, Omega_2B = get_operator_from_y(y, dim1B, dim2B)
  #print(Omega_2B) 
  
  #Transform Hamiltonian using Omega
  Transformed_Hamiltonian=Transformed_Ham(Omega_1B, Omega_2B, user_data)
  
  # calculate the generator
  Eta_1B, Eta_2B = calc_eta(Transformed_Hamiltonian[1], Transformed_Hamiltonian[2], user_data)
  
  # calculate the right-hand side
  dOmega_0B, dOmega_1B, dOmega_2B = calc_rhs(Omega_1B,Omega_2B,Eta_1B,Eta_2B, user_data)
  
  # convert derivatives into linear array
  dy   = np.append([dOmega_0B], np.append(reshape(dOmega_1B, -1), reshape(dOmega_2B, -1)))
  # share data
  user_data["dE"] = dOmega_0B #storing dOmega_0B/ds in dE
  user_data["eta_norm"] = np.linalg.norm(Eta_1B,ord='fro')+np.linalg.norm(Eta_2B,ord='fro')
  
  return dy

#-----------------------------------------------------------------------------------
# pairing Hamiltonian
#-----------------------------------------------------------------------------------
def pairing_hamiltonian(delta, g, user_data):
  bas1B = user_data["bas1B"]
  bas2B = user_data["bas2B"]
  idx2B = user_data["idx2B"]

  dim = len(bas1B)
  H1B = np.zeros((dim,dim))

  for i in bas1B:
    H1B[i,i] = delta*np.floor_divide(i, 2)

  dim = len(bas2B)
  H2B = np.zeros((dim, dim))

  # spin up states have even indices, spin down the next odd index
  for (i, j) in bas2B:
    if (i % 2 == 0 and j == i+1):
      for (k, l) in bas2B:
        if (k % 2 == 0 and l == k+1):
          H2B[idx2B[(i,j)],idx2B[(k,l)]] = -0.5*g
          H2B[idx2B[(j,i)],idx2B[(k,l)]] = 0.5*g
          H2B[idx2B[(i,j)],idx2B[(l,k)]] = 0.5*g
          H2B[idx2B[(j,i)],idx2B[(l,k)]] = -0.5*g
  
  return H1B, H2B

#-----------------------------------------------------------------------------------
# normal-ordered pairing Hamiltonian
#-----------------------------------------------------------------------------------
def normal_order(H1B, H2B, user_data):
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # 0B part
  E = 0.0
  for i in holes:
    E += H1B[i,i]

  for i in holes:
    for j in holes:
      E += 0.5*H2B[idx2B[(i,j)],idx2B[(i,j)]]  

  # 1B part
  f = H1B
  for i in bas1B:
    for j in bas1B:
      for h in holes:
        f[i,j] += H2B[idx2B[(i,h)],idx2B[(j,h)]]  

  # 2B part
  Gamma = H2B

  return E, f, Gamma

#-----------------------------------------------------------------------------------
# Perturbation theory
#-----------------------------------------------------------------------------------
def calc_mbpt2(f, Gamma, user_data):
  DE2 = 0.0

  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  for i in holes:
    for j in holes:
      for a in particles:
        for b in particles:
          denom = f[i,i] + f[j,j] - f[a,a] - f[b,b]
          me    = Gamma[idx2B[(a,b)],idx2B[(i,j)]]
          DE2  += 0.25*me*me/denom

  return DE2

def calc_mbpt3(f, Gamma, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # DE3 = 0.0

  DE3pp = 0.0
  DE3hh = 0.0
  DE3ph = 0.0

  for a in particles:
    for b in particles:
      for c in particles:
        for d in particles:
          for i in holes:
            for j in holes:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[i,i] + f[j,j] - f[c,c] - f[d,d])
              me    = Gamma[idx2B[(i,j)],idx2B[(a,b)]]*Gamma[idx2B[(a,b)],idx2B[(c,d)]]*Gamma[idx2B[(c,d)],idx2B[(i,j)]]
              DE3pp += 0.125*me/denom

  for i in holes:
    for j in holes:
      for k in holes:
        for l in holes:
          for a in particles:
            for b in particles:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[l,l] - f[a,a] - f[b,b])
              me    = Gamma[idx2B[(a,b)],idx2B[(k,l)]]*Gamma[idx2B[(k,l)],idx2B[(i,j)]]*Gamma[idx2B[(i,j)],idx2B[(a,b)]]
              DE3hh += 0.125*me/denom

  for i in holes:
    for j in holes:
      for k in holes:
        for a in particles:
          for b in particles:
            for c in particles:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[j,j] - f[a,a] - f[c,c])
              me    = Gamma[idx2B[(i,j)],idx2B[(a,b)]]*Gamma[idx2B[(k,b)],idx2B[(i,c)]]*Gamma[idx2B[(a,c)],idx2B[(k,j)]]
              DE3ph -= me/denom
  return DE3pp+DE3hh+DE3ph

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

def main():
  # grab delta and g from the command line
  delta      =1
  g          = 0.5
  # Reference state:
  holes=[0,1,2,3]
  particles=[4,5,6,7]
  # setup shared data
  dim1B = len(holes)+len(particles)
  dim2B = dim1B*dim1B
  # basis definitions
  bas1B     = range(dim1B)
  bas2B     = construct_basis_2B(holes, particles)
  basph2B   = construct_basis_ph2B(holes, particles)

  idx2B     = construct_index_2B(bas2B)
  idxph2B   = construct_index_2B(basph2B)

  # occupation number matrices
  occ1B     = construct_occupation_1B(bas1B, holes, particles)
  occA_2B   = construct_occupationA_2B(bas2B, occ1B)
  occB_2B   = construct_occupationB_2B(bas2B, occ1B)
  occC_2B   = construct_occupationC_2B(bas2B, occ1B)

  occphA_2B = construct_occupationA_2B(basph2B, occ1B)
  bn2=Bernoulli_generator(16)# Go up to 16 terms in expansion

  # store shared data in a dictionary, so we can avoid passing the basis
  # lookups etc. as separate parameters all the time
  user_data  = {
    "dim1B":      dim1B, 
    "holes":      holes,
    "particles":  particles,
    "bas1B":      bas1B,
    "bas2B":      bas2B,
    "basph2B":    basph2B,
    "idx2B":      idx2B,
    "idxph2B":    idxph2B,
    "occ1B":      occ1B,
    "occA_2B":    occA_2B,
    "occB_2B":    occB_2B,
    "occC_2B":    occC_2B,
    "occphA_2B":  occphA_2B,
    "bn2": bn2,
    # variables for sharing data between ODE solver
    "eta_norm":   0.0,                #Initial norm of eta
    "dE":         0.0,                # Initial dOmega/ds
    "calc_eta":   eta_white,          # specify the generator (function object)
    "calc_rhs":   RHS_Cal,         # specify the right-hand side and truncation
    "special_commutator": special_commutator #commutator used to evaluate n body commuatators: 0 up to 2 body matrices
  }

  # set up initial Hamiltonian
  H1B, H2B = pairing_hamiltonian(delta, g, user_data)

  E, f, Gamma = normal_order(H1B, H2B, user_data)#initial Hamiltonian
  
  # Append Initial Hamiltonian to dictionary since that won't be changing in the Magnus expansion
  user_data["E"]= E
  user_data["f"]= f
  user_data["Gamma"]= Gamma
  
  #Initializations
  Initial_Omega0=0.
  Initial_Omega1=np.zeros((dim1B,dim1B))
  Initial_Omega2=np.zeros((dim1B**2,dim1B**2))
  
  # reshape generator into a linear array (initial ODE vector)
  y0   = np.append([Initial_Omega0], np.append(reshape(Initial_Omega1, -1), reshape(Initial_Omega2, -1)))

  # integrate flow equations 
  solver = ode(derivative_wrapper,jac=None)
  solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
  solver.set_f_params(user_data)
  solver.set_initial_value(y0, 0.)

  sfinal = 50
  ds = 0.1

  print ("%-20s   %-28s   %-20s   %-20s   %-20s   %-14s   %-25s   %-16s   %-8s"%(
    "s", "E" , "DE(2)", "DE(3)", "E+DE", "dOmega_0B/ds", 
    "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print ("-" * 185)
  while solver.successful() and solver.t < sfinal:
    ys = solver.integrate(sfinal, step=True)
    
    Omega_0B,Omega_1B, Omega_2B = get_operator_from_y(ys, dim1B, dim2B)#get Omegas
    E, f , Gamma=Transformed_Ham(Omega_1B, Omega_2B, user_data)#get  Hamiltonian to use MBPT, and print out values
    
    energy.append(E)#append energy for plotting
    flow.append(solver.t)#append flow parameter for plotting


    DE2 = calc_mbpt2(f, Gamma, user_data)
    DE3 = calc_mbpt3(f, Gamma, user_data)

    norm_fod     = calc_fod_norm(f, user_data)
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)

    print ("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
        solver.t, E , DE2, DE3, E+DE2+DE3, user_data["dE"], user_data["eta_norm"], norm_fod, norm_Gammaod))
    if abs(DE2/E) < 10e-8: break


    solver.integrate(solver.t + ds)

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()
plt.plot(flow,energy)