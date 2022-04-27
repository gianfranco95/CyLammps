#!/usr/bin/python3

import sys
import re
import numpy as np
from libreria import timestep
from multiprocessing import Pool

#IMPORT DATA****************************************

files=sys.argv[1:]
files.sort(key=timestep)
dx=1 			#bin_x
Nt = len(files) 
#***************************************************

def TOPOGRAFIA(j):
	data = np.loadtxt(files[j],usecols=(2,3,4),skiprows=9)
	LX =  2*np.loadtxt(files[j],usecols=(1),skiprows=5,max_rows=1)
	Ly =  2*np.loadtxt(files[j],usecols=(1),skiprows=6,max_rows=1)
	step = int(timestep(files[j])/10**4)

	#CREAZIONE PROFILO**********************************
	x= data[:,0]
	y= data[:,1] - Ly*np.around(data[:,1]/(Ly))
	z= data[:,2]

	xmin = x.min()
	ymin = y.min()

	Lx=x.max()-xmin				
	N = len(x)						#numero pallette
	dy= 2							#bin_y
	Nx = int(np.floor(Lx/dx))		#numero di bin lungo x
	Ny = int(np.floor(Ly/dy))		#numero di bin lungo y 
	N_M= int(dx)*dy +1				#numero di massimi da registrare
	#Ncut = 2
	#X = X[Ncut:-Ncut]
	z= z - z.min() + 0.000001

	#Profilo = np.zeros((Nx-2*Ncut,Ny,N_M))
	Profilo = np.zeros((Nx,Ny,N_M))

	for i in range(N):
		a = int(np.floor((x[i]-xmin)/dx))
		b = int(np.floor((y[i]-ymin)/dy))
		
		if(a<Nx and b<Ny):
			m = Profilo[a,b,:].min()
			if (z[i]>m):
				k, = np.where(Profilo[a,b,:]==m)
				Profilo[a,b,k[0]] = z[i]

	Profilo = Profilo.mean(axis=2)
	Profilo = Profilo.mean(axis=1)

	L = np.zeros(len(Profilo))
	L[0] = LX	#box
	L[1] = Lx	#sistema
	
	np.savetxt('profilo/Profilo_{:d}.dat'.format(step),np.c_[L,Profilo])
	
	return;


with Pool(60) as pool:
	pool.map(TOPOGRAFIA,np.arange(Nt))
