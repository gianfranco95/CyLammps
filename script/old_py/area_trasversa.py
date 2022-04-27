#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
from multiprocessing import Pool
from persistence1d import RunPersistence
import scipy.integrate as integrate

def lista(r,N):
	val= []
	for i in range(N):
		val.append((i,r))
	return val

def FilterExtremaByPersistence(ExtremaAndPersistence, Threshhold):
	FilteredExtremaAndPersistence = [t for t in ExtremaAndPersistence if t[1] > Threshhold]
	return FilteredExtremaAndPersistence

#*******************************************************************************************************************************
NMAX=7		#cambiare con L0->  7,9,13,17
cut = 10           #lo stesso usato per calcolare l'ampiezza e per k
L0 = 200	#cambiare

def func(i,run):
    DIR = "/ddn/leporini/gianfranco/box_"+str(int(L0))+"_13_9/deform/run"+str(int(run))+"/"
    data = np.loadtxt(DIR + "lmp_data/Conf_"+ str(int(i*40000)) +".dat",skiprows=9,usecols=(2,4))   #cambiare con L0 -> 40k,60k,100k,140k
    profilo = np.loadtxt(DIR + 'profilo/Profilo_'+str(int(i*4))+'.dat',usecols=(1))			#cambiare con L0 -> 4,6,10,14
    x = np.arange(len(profilo))

    ExtremaAndPersistence = RunPersistence(profilo[cut:-cut])
    ExtremaAndPers = FilterExtremaByPersistence(ExtremaAndPersistence,0.5)
    if(len(ExtremaAndPers)<4):
        a=0
        while(len(ExtremaAndPers)<4):
            a=a+1
            ExtremaAndPers = FilterExtremaByPersistence(ExtremaAndPersistence,1-a*0.05)

    IdEx = np.array([t[0] for t in ExtremaAndPers])
    IdEx = np.sort(IdEx[::2])+cut

    xpoint = IdEx
#    xpoint = np.append(IdEx,len(profilo)-1)
#    xpoint = np.insert(xpoint,0,0)
    Point = profilo[IdEx]
#    Point = np.append(profilo[IdEx],profilo[-1])
#    Point = np.insert(Point,0,profilo[0])

    f = interp1d(xpoint,Point)
    X = np.arange(xpoint[0],xpoint[-1]+1,1)

    J1 = integrate.simpson(profilo[xpoint[0]:xpoint[-1]],x[xpoint[0]:xpoint[-1]])
    J2 = integrate.simpson(Point,xpoint)
    J = J1-J2

    Z = data[:,1] - data[:,1].min()
    xmin = data[:,0].min()
    I = (np.floor(np.copy(data[:,0]) - xmin)).astype(int)
    I1 =I[I!=I.max()]

#    boolWrink = Z[I!=I.max()] > f(I1)
    boolI = (I >= xpoint[0]) & (I <= xpoint[-1])
    boolWrink = Z[boolI] > f(I[boolI])

    nwrink = boolWrink[boolWrink==True].shape[0]
    return [i,nwrink,J]


Nwrink = []		#numero particelle nella increspatura
RUN = np.concatenate( (np.arange(1,22),np.arange(23,71)) )

for r in RUN:
	values = lista(r,51)
	with Pool(60) as pool:
		res = pool.starmap(func,values)
	res = np.array(res)
	np.sort(res.view("i8,i8,i8"), order=['f0'],axis=0).view(int)
	Nwrink.append(res)

Nwrink = np.array(Nwrink)
Nmean = np.mean(Nwrink[:,:,1],axis=0)
Nstd = np.std(Nwrink[:,:,1],axis=0)/np.sqrt(70)

Jmean = np.mean(Nwrink[:,:,2],axis=0)
Jstd = np.std(Nwrink[:,:,2],axis=0)/np.sqrt(70)

np.savetxt("/ddn/leporini/gianfranco/box_"+str(int(L0))+"_13_9/deform/Popolazione_wrinkling_senza.dat",np.c_[np.arange(51),Nmean,Nstd,Jmean,Jstd] )
