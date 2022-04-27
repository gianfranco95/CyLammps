#!/bigdisk/cordella/anaconda3/envs/py_gian/bin/python

from glob import glob
import numpy as np
import psutil
import ray
import sys  
from libreria import initialize_bin



pipeline = glob('lmp_data/Conf_*.bin')

res = []
[res.append(initialize_bin.remote(file)) for file in pipeline]

X,box,step,Npar = zip(*ray.get(res ) )

X = np.array(X)
box = np.array(box)
bolstep = np.argsort( np.array(step) )


X = X[bolstep]
box = box[bolstep]


def wrap(periodX=True,periodY=True,periodZ=True):
	if periodX == True :
		Lx = 2*box[:,0][:,None]
		X[:,:,0] = X[:,:,0] - Lx*np.around(X[:,:,0]/Lx)
	if periodY == True :
		Ly = 2*box[:,1][:,None]
		X[:,:,1] = X[:,:,1] - Ly*np.around(X[:,:,1]/Ly)
	if periodZ == True :
		Lz = 2*box[:,2][:,None]
		X[:,:,2] = X[:,:,2] - Lz*np.around(X[:,:,2]/Lz)


wrap(periodZ=False)


@ray.remote
def Cee(X,box,t):                                                                                       #m e' il # di monomeri di una catena
	m=3
	Xee = X[:,m-1::m,:] - X[:,::m,:]
	Xee[:,:,0] = Xee[:,:,0] - 2*box[:,0][:,None]*np.around(Xee[:,:,0]/(2*box[:,0][:,None]))
	Xee[:,:,1] = Xee[:,:,1] - 2*box[:,1][:,None]*np.around(Xee[:,:,1]/(2*box[:,1][:,None]))

	cee0 = np.sum(Xee*Xee, axis=2).mean()
	cee =  np.sum(Xee[t:]*Xee[:-t], axis=2).mean()           #la prima media e' sui tempi la seconda sulle molecole
	cee = cee/cee0                                          #normalizzo rispetto al valore iniziale

	return [t,cee]


print('ora pool')



Xid  = ray.put(X)
Boxid  = ray.put(box)

res = ray.get([Cee.remote(Xid,Boxid, j) for j in range(1,len(pipeline)-2)])

res = np.array(res)

bolord = np.argsort(res[:,0])

res = res[bolord]

np.savetxt('Cee.dat',res)
