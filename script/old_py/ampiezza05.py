#!/usr/bin/python3

from glob import glob
from libreria import timestep
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from multiprocessing import Pool
from persistence1d import RunPersistence

LL0=200 	#cambiare
DIR1 = "/ddn/leporini/gianfranco/box_" + str(int(LL0))+"_13_9/"

def lista(a,b,L0='None'):
	val= []
	for i in range(a,b):
        	val.append((L0,i))
	return val


def FilterExtremaByPersistence(ExtremaAndPersistence, Threshhold):
    FilteredExtremaAndPersistence = [t for t in ExtremaAndPersistence if t[1] > Threshhold]
    return FilteredExtremaAndPersistence

def IdxExtrema(ExtremaAndPersistence):
    IDXextr = [t[0] for t in ExtremaAndPersistence]
    return IDXextr.sort()

def pressione_pistone(L0,Idx):
    AMP = []
    for f in range(51):
        profilo = np.loadtxt(DIR1 + "deform/run{:d}/profilo/Profilo_{:d}.dat".format(Idx,int(f*4)))  #cambiare con L0

        "CALCOLO AMPIEZZA"
        profilo = profilo[10:-10,1]
        ExtremaAndPersistence = RunPersistence(profilo)
        ExtremaAndPers = FilterExtremaByPersistence(ExtremaAndPersistence,0.5)

        if(len(ExtremaAndPers)<4):
            a=0
            while(len(ExtremaAndPers)<4):
                a=a+1
                ExtremaAndPers = FilterExtremaByPersistence(ExtremaAndPersistence,1-a*0.05)

        IdEx = sorted([t[0] for t in ExtremaAndPers])
        ampiezza = [ (2*profilo[IdEx[i+1]]-profilo[IdEx[i+2]]- profilo[IdEx[i]])/2 for i in range(len(IdEx)-2)  ]
        ampiezza=np.abs(np.array(ampiezza)).mean()

        AMP.append(ampiezza)

    AMP = np.array(AMP)

    np.savetxt(DIR1 + "deform/run{:d}/ampiezza05.dat".format(Idx),np.c_[np.arange(51),AMP])

    return


def main():
	values= lista(1,22,L0=LL0)        #cambiare con L0
	with Pool(71) as pool:
		pool.starmap(pressione_pistone,values)



if __name__ == '__main__':
    main()
