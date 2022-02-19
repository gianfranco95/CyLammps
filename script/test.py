from ANALISI import CGSTRESS
from ANALISI import PROFILO
from ANALISI import CORR_STRESS
from ANALISI import MAPPA_H
from ANALISI import VOLUME

from timeit import default_timer as timer
from multiprocessing import Pool
import numpy as np
import glob


def func(curva):
    curva[:,0] = (curva[0,0]-curva[:,0])/curva[0,0]
    Idrop = np.arange(curva.shape[0]-1)[curva[1:,7] - curva[:-1,7] < - 1e-5  ] +1
    edrop = curva[Idrop,0]
    drops = np.abs(curva[Idrop,7] - curva[Idrop-1,7])
    return Idrop


#Np numero di processori da usare per run in parallelo
def do_profilo(Idrop,Np):
	#calcola sia Hxy che densita'
	val = np.array([ [i-1,i] for i in Idrop]).ravel().astype(np.int_)
	with Pool(Np) as pool:
		res = pool.map(PROFILO,val)
	return

def do_CGstress(Idrop,Np):
	#calcola solo stress CG su griglia 3d
	val = np.array([ [i-1,i] for i in Idrop]).ravel().astype(np.int_)
	with Pool(Np) as pool:
		pool.map(CGSTRESS,val)
	return

def do_corrStress(t):
	CORR_STRESS(t)
	return

def do_Hx_strain(Idrop,Nx,Np):
	#calcolata solo sugli eventi plastici
	val = []
	for a in Idrop:
		val.append([a,Nx])

	with Pool(Np) as pool:
		result = pool.starmap(MAPPA_H,val)

	tt = np.array([result[i][0] for i in range(len(result)) ])
	H  = np.array([result[i][1] for i in range(len(result)) ])

	boll = np.argsort(tt)
	H = H[boll]
	np.save(f'Hx_strain_grid.npy' , H, allow_pickle=False,fix_imports=False)
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





####################################################################################################################
data = np.loadtxt('../curva_carico.dat')
Idrop = func(data)


start = timer()

do_volume(80)

end = timer()

print(f'tempo {end-start}')
