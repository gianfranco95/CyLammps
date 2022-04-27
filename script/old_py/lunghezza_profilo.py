#!/usr/bin/python3

import numpy as np

DIR = "/ddn/leporini/gianfranco/"
RUN = np.concatenate((np.arange(1,22),np.arange(23,71)))
#RUN=np.arange(1,71)

def LUNGH_CORRUG(L):
    LUNGHEZZA = []
    for r in RUN:
        lung = []
        for d in range(0,51):
            profilo = np.loadtxt( DIR + "box_"+str(L)+"_13_9/deform/run"+str(r)+"/profilo/Profilo_"+ str(int(d*2*(L/100)))+".dat")
            l=0
            for j in range(1,len(profilo)):
                l= l + np.sqrt((profilo[j,1]-profilo[j-1,1])**2+1)
            lung.append([l])
        lung=np.array(lung)
        LUNGHEZZA.append(lung)

    LUNGHEZZA = np.array(LUNGHEZZA)
    DLUNGHEZZA = np.std(LUNGHEZZA,axis=0)/np.sqrt(len(RUN))
    LUNGHEZZA = LUNGHEZZA.mean(axis=0)

    np.savetxt(DIR + "box_"+str(L)+"_13_9/deform/medie/lunghezza_corrugazione.dat",np.c_[np.arange(51),LUNGHEZZA,DLUNGHEZZA])

    return


L0 = [200,300,500,700]

LUNGH_CORRUG(200)
