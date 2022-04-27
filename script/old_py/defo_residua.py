#!/usr/bin/python2.7

import numpy as np
from libreria import timestep
from glob import glob

Lresidua=[]

for i in range(1,9):
	files = glob('run{:d}/lmp_data/*'.format(i))
	data =[]
	Lbox=[]
	for f in files:
		data.append(np.loadtxt('{:s}'.format(f),usecols=(2),skiprows=9))
		Lbox.append(2*np.loadtxt('{:s}'.format(f),usecols=(1),skiprows=5,max_rows=1))

	data = np.array(data)
	Lbox = np.array(Lbox)
	Lc = Lbox.min()
	L = (data.max(axis=1) - data.min(axis=1))
	
	Ltemp=[]
	for i in range(len(L)):
		if( Lbox[i]-L[i] <3):
			Ltemp.append(L[i])

	Lresidua.append( [Lc,np.array(Ltemp).max()] )

Lresidua = np.array(Lresidua).mean(axis=0)
np.savetxt('defo_res.dat',np.c_[Lresidua])
