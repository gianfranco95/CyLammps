from timeit import default_timer as timer
from multiprocessing import Pool
import glob
import re

import numpy as np

from ANALISI import CGSTRESS
from ANALISI import PROFILO
from ANALISI import CORR_STRESS
from ANALISI import MAPPA_H
from ANALISI import MAPPA_DH
from ANALISI import VOLUME


def last_stress(path):
	count=[]
	for f in path:
		count.append( int(re.findall(r'\d+', f)[-1]) )
	count = np.array(count)
	return count.max()

def func(curva):
    curva[:,0] = (curva[0,0]-curva[:,0])/curva[0,0]
    Idrop = np.arange(curva.shape[0]-1)[curva[1:,7] - curva[:-1,7] < - 1e-8  ] +1
    edrop = curva[Idrop,0]
    drops = np.abs(curva[Idrop,7] - curva[Idrop-1,7])
    return edrop, Idrop


#Np numero di processori da usare per run in parallelo
def do_profilo(Idrop,Np):
	#calcola sia Hxy che densita'
#	val = np.array([ [i-1,i] for i in Idrop]).ravel().astype(np.int_)
	with Pool(Np) as pool:
		res = pool.map(PROFILO,Idrop)
	return

def do_CGstress(Idrop,Np):
	#calcola solo stress CG su griglia 3d
#	val = np.array([ [i-1,i] for i in Idrop]).ravel().astype(np.int_)
	with Pool(Np) as pool:
		pool.map(CGSTRESS,Idrop)
	return

def do_corrStress(val,Np):
	#calcola la matrice di autocorrelazione dello stress 2d
	with Pool(Np) as pool:
		pool.map(CORR_STRESS, val)
	return

def do_Hx_strain(Idrop,Nx,Np):
	#calcolata solo sugli eventi plastici mappa H(x,epsilon)
	val = []
	for a in Idrop:
		val.append([a,Nx])

	with Pool(Np) as pool:
		result = pool.starmap(MAPPA_H,val)

	tt = np.array([result[i][0] for i in range(len(result)) ])
	H  = np.array([result[i][1] for i in range(len(result)) ])

	boll = np.argsort(tt)
	H = H[boll]
	np.save(f'Hx_ALLstrain_grid.npy' , H, allow_pickle=False,fix_imports=False)
	return


def do_volume(Np):
	#calcola il volume totale del sistema
	Nt = len( glob.glob('../lmp_data/Conf*.bin'))
	with Pool(Np) as pool:
		res = pool.map(VOLUME,np.arange(Nt).astype(np.int_))

	time = [res[j][0] for j in range(Nt)]
	Vol = [res[j][1] for j in range(Nt)]

	time = np.array(time)
	Vol = np.array(Vol)

	bol = np.argsort(time)
	Vol = Vol[bol]

	np.savetxt('volume.dat',Vol)
	return




def do_DHx_eps(eps,Nx,Np,Emax):
	'''
	calcolata solo sugli eventi plastici mappa H(x,epsilon) vedendo gli spostamenti delle particelle in superficie....
	Emax e' la deformazione massima
	'''
	bins = []
	for a in range(int(Emax/0.0005)):
		j = np.argsort(np.abs(eps-0.0005*a))[0]
		if j < eps.shape[0]-1:
			bins.append(j)
	bins = np.array(bins)

	val = []
	for a in range(len(bins)-1):
		val.append([bins[a],bins[a+1],Nx])

	with Pool(Np) as pool:
		result = pool.starmap(MAPPA_DH,val)

	tt = np.array([result[i][0] for i in range(len(result)) ])
	DH  = np.array([result[i][1] for i in range(len(result)) ])

	boll = np.argsort(tt)
	DH = DH[boll]
	DH = np.c_[ eps[bins[1:]], DH]
	np.save(f'DHx_grid.npy' , DH, allow_pickle=False,fix_imports=False)
	return


##################################################################################################################
def main():
	data = np.loadtxt('../curva_carico.dat')
	edrop,Idrop = func(data)

	Nlast = last_stress( glob.glob('stress/stress_*.npy') )

	bol = edrop < 0.15
#	bol = Idrop > Nlast
	Idrop = Idrop[bol]
#	print(Idrop.shape)

	start = timer()

	val = np.arange(Idrop[-1])
	print(val.shape)
#	do_DHx_eps(data[:,0],2*500,90,edrop[-1])
#	do_volume(20)

	Np = min(val.shape[0],85)
	do_profilo(val,Np)
	do_CGstress(val,Np)
	#do_corrStress(Idrop,90)

	end = timer()
	print(f'tempo {end-start}')


if __name__ == '__main__':
        main()
