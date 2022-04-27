#!/usr/bin/python3
import sys
import re
import numpy as np
from libreria import timestep
from libreria import butter
from libreria import butter_filter
from scipy.signal import get_window
from scipy.signal import welch
from multiprocessing import Pool

files=sys.argv[1:]
files.sort(key=timestep)
N = len(files)
dx = 1
data = []

for f in files:
    data.append(np.loadtxt(f))

window = 'boxcar'

def SPETTRO(j):
	Lbox = data[j][0,0]	
	Leff = data[j][1,0]
	step = int(timestep(files[j]))
	Profilo = data[j][10:-10,1]
	Profilo = Profilo - np.mean(Profilo)

	#Profilo = butter_filter(Profilo,Kcut,1/dx,'low')
	#Profilo = butter_filter(Profilo, 4.875/(10*dx*len(Profilo)),1/dx,'high')   #risoluzione kaiser 
	#Profilo = butter_filter(Profilo, 1/Leff,1/dx,'high')
	#FFT********************************************************
	Nz= int(len(Profilo)*2)						#numero totale sample contenenti anche zeri 

	PROFILO = Profilo*get_window(window,Profilo.size)
	sp = np.fft.rfft( PROFILO , n=Nz)
	freq = np.fft.rfftfreq(Nz,d=dx)
	Potenza = np.abs(sp)**2
	Potenza = Potenza[1:]
	freq = freq[1:]

	kmax = [ Leff, freq[np.where(Potenza==Potenza.max())[0][0] ] ]
	L = np.zeros(len(freq))
	L[0] = Lbox		#box
	L[1] = Leff		#sistema
#	np.savetxt('fft/fft_{:d}.dat'.format(step),np.c_[freq,Potenza,L])
	return kmax;

#Calcolo dello spettro

with Pool(60) as pool:
	KMAX =  pool.map(SPETTRO,np.arange(N)[::2]) 

KMAX = np.array(KMAX)
Kmaxsort = KMAX[np.argsort(KMAX[:, 0])][::-1]


np.savetxt('kmax.dat',Kmaxsort)


