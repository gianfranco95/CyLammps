#!/usr/bin/python2.7

import sys
import re
import numpy as np
from libreria import timestep
#IMPORT DATA****************************************

files=sys.argv[2:]
files.sort(key=timestep)

dx=float(sys.argv[1])			#bin_x

Nt = len(files) 
#***************************************************

def TOPOGRAFIA(j):
	data = np.array([np.loadtxt(files[j],usecols=(2,3,4),skiprows=9)])
	step = int(timestep(files[j]))
	LX =  2*np.loadtxt(files[j],usecols=(1),skiprows=5,max_rows=1)
	Ly =  2*np.loadtxt(files[j],usecols=(1),skiprows=6,max_rows=1)

	#CREAZIONE PROFILO**********************************
	x= data[0,:,0]
	y= data[0,:,1] - Ly*np.around(data[0,:,1]/(Ly))
	z= data[0,:,2]

	Lx=x.max()-x.min()
	N = len(x)						#numero pallette
	dy= 2							#bin_y
	Nx = int(np.floor(Lx)/dx)    		#numero di bin lungo x
	Ny = int(np.floor(Ly)/dy)	        #numero di bin lungo y 
	N_M= int(dx)*dy +1			#numero di massimi da registrare
	X = np.arange(1,Nx+1)*dx - np.floor(Lx)/2
	Y = np.arange(1,Ny+1)*dy - np.floor(Ly)/2
	
	z= z - z.min() + 0.000001

	Profilo = np.zeros((Nx,Ny,N_M))

	for i in range(N):
		aa=0
		bb=0
		
		if ( (np.floor((X-x[i])/dx)==0).any()  ):
			a, = np.where(np.floor((X-x[i])/dx)==0)
		else:
			aa = 1

		if ( (np.floor((Y-y[i])/dy)==0).any() ):		
			b, = np.where(np.floor((Y-y[i])/dy)==0)
		else:	
			bb=1


		if(aa==0 and bb==0):
			m = np.min(Profilo[a[0],b[0],:])

			if (Profilo[a[0],b[0],:].any() == 0 ):
				k, = np.where(Profilo[a[0],b[0],:]==0)
				Profilo[a[0],b[0],k[0]]= z[i]
			elif (z[i] > m):
				k, = np.where(Profilo[a[0],b[0],:]==m)
				Profilo[a[0],b[0],k[0]] = z[i]

		


	Profilo = np.mean(Profilo[:,1:], axis=2)

	Profilo = Profilo.mean(axis=1)

	L = np.zeros(len(Profilo))
	L[0] = LX	#box
	L[1] = Lx	#sistema
	
	np.savetxt('profilo/Profilo_{:d}.dat'.format(step),np.c_[L,Profilo])
	
	return;


for i in range(Nt):
	TOPOGRAFIA(i)
