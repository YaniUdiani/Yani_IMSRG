#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:06:55 2020

@author: YaniUdiani
"""

import matplotlib.pyplot as plt
import pylab
num_sp = [38, 54, 66, 114, 162, 186, 246, 294, 342, 358]
jump = []

for i in range(len(num_sp)-1):
  jump.append(num_sp[i+1]/num_sp[i])
  
print(jump) 
#Nmax = [1,2,3,4,5,6,7,8,9,10]
real_minnesota = [0.6702508960573477,0.6426666666666667, 0.6666666666666666, 0.6979472140762464,
                  0.7216494845360825, 0.7486818980667839, 0.7720144752714113 , 0.7703109327983951, 0.7591763652641003, 0.7638533674339301]
coe = [-0.1549, -0.4005, -0.4113, -0.6809, -1.0042, -1.0416, -1.0823, -1.111, -1.1225, -1.123]
de2_zero = [-1.92303298, -5.13860274, -5.32987980, -9.24590334, -13.78366919,-14.40589494,-15.11699426 ,-15.61488783, -15.82963608, -15.84543077] 
mod_minnesota = [0.8602150537634409, 0.736, 0.7243243243243244, 0.7272727272727273,
                 0.7360824742268042, 0.7680140597539543, 0.7961399276236429, 0.7873620862587764, 0.7645478961504029,0.7698209718670077]


check = [0.2459809571753624, 0.468859995753121, 0.428913444931311, 0.28665790076388475, 0.2757368894750982,
         0.1759198321549036, 0.035293235286530944, -0.060722662083321666, -0.11407635230250435, -0.12310083851633635]

plt.plot(num_sp, real_minnesota, label ="Original")
plt.plot(num_sp, mod_minnesota, label ="spin-off")
plt.xlabel("Number of sp states")
plt.ylabel("Fraction of Zero Blks")
plt.legend()

plt.figure()
plt.plot(num_sp, coe)
plt.xlabel("Number of sp states")
plt.ylabel("Correlation energy")

plt.figure()
new_coe = coe[0:9]
plt.plot(jump, new_coe)
plt.xlabel("Jumps in Number of sp states")
plt.ylabel("Correlation energy")
