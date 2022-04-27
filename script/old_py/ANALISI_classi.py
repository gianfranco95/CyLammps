#!/usr/bin/python3

import re
from glob import glob
import numpy as np
import sympy as smp
from scipy import spatial
from scipy.spatial import KDTree
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import gradient
from timeit import default_timer as timer
from multiprocessing import Pool


def initialize_bin(file):
    step, Npar, tricl,boundary, box,Nfield,Nproc = zip(*np.fromfile(file,dtype=np.dtype('i8,i8,i4,6i4,6f8,i4,i4'), count =1))
    step=step[0]
    Npar = Npar[0]
    boundary = boundary[0]
    box = box[0]
    Nfield = Nfield[0]
    Nproc = Nproc[0]
    X = np.fromfile(file,dtype=np.dtype(f'i4,{int(Nfield*Npar/Nproc)}f8'), count = Nproc,offset=100)
    X = np.array([X[i][1].reshape(int(Npar/Nproc),Nfield) for i in range(Nproc)])

    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2] )

    return X,box,step,boundary,Npar

def read_lammps(filename,n_proc):
#    path = glob(filename)
    path = filename

    with Pool(n_proc) as pool:
        X,box,step,boundary,Npar = zip(*pool.map(initialize_bin,path) )

    t = np.argsort(np.array(step))
    X = np.array(X)[t]
    box = np.array(box)[t]
    step = np.array(step)[t]
    boundary = boundary[0]
    Npar = Npar[0]
    return X,box,step,boundary,Npar


#classe particelle*************************************************************************************************************************************************** 
class Particles:
	def __init__(self,path,n_proc,m,epsilon,sigma,k,l0) :
		self._X, self._box, self._step, self._boundary, self._Npar = read_lammps(path,n_proc)
		self._m = m
		self._epsilon = epsilon
		self._sigma = sigma
		self._k = k
		self._l0 = l0
		self._auto_wrap()

	@property
	def X(self): 					#coordinate particelle (len tempi, # particelle, 3)
		return self._X
	@property
	def box(self): 					#box con dimensioni della scatola  (len tempi,6)
		return self._box
	@property
	def boundary(self):				#condizioni al bordo al tempo iniziale (6,)
		return self._boundary
	@property
	def Npar(self): 				#numero di particelle
		return self._Npar
	@property
	def step(self): 				#array dei tempi (len tempi)
		return self._step
	@property
	def m(self): 					#numero di monomeri per catena
		return self._m

	@property
	def epsilon(self): 					#numero di monomeri per catena
		return self._epsilon
	@property
	def sigma(self):
		return self._sigma
	@property
	def k(self):
		return self._k
	@property
	def l0(self):
		return self._l0


	def _auto_wrap(self):					#wrappo le coordinate e riporto il box tra 0 ed L
		if (self.boundary[0]==0):
			Lx = (self.box[:,1] -self.box[:,0])[:,None]
			self._X[:,:,0] = self._X[:,:,0] - Lx*np.around(self._X[:,:,0]/Lx) + Lx/2
		if (self.boundary[2]==0):
			Ly = (self.box[:,3] -self.box[:,2])[:,None]
			self._X[:,:,1] = self._X[:,:,1] - Ly*np.around(self._X[:,:,1]/Ly) + Ly/2
		if (self.boundary[4]==0):
			Lz = (self.box[:,5] -self.box[:,4])[:,None]
			self._X[:,:,2] = self._X[:,:,2] - Lz*np.around(self._X[:,:,2]/Lz) + Lz/2
		else:
			self._X[:,:,2] = self._X[:,:,2] - self.box[:,4][:,None]


	def wrap(self,t,R):							#metodo per wrappare array a caso, funziona sia con griglie 3d shape(M,M,3) che array 2d shape(M,3)
		if (self.boundary[0]==0):
			Lx = (self.box[t,1] -self.box[t,0])
			R[...,0] =  R[...,0] - Lx*np.around(R[...,0]/Lx)
		if (self.boundary[2]==0):
			Ly = (self.box[t,3] -self.box[t,2])
			R[...,1] =  R[...,1] - Ly*np.around(R[...,1]/Ly)
		if (self.boundary[4]==0):
			Lz = (self.box[t,5] -self.box[t,4])
			R[...,2] =  R[...,2] - Lz*np.around(R[...,2]/Lz)
		return R

	def Cee(self,m):                              #m e' il # di monomeri di una catena
		Lx = (self.box[:,1] -self.box[:,0])[:,None]
		Ly = (self.box[:,3] -self.box[:,2])[:,None]

		Xee = self.X[:,m-1::m,:] - self.X[:,::m,:]
		Xee[:,:,0] = Xee[:,:,0] - Lx*np.around(Xee[:,:,0]/Lx)
		Xee[:,:,1] = Xee[:,:,1] - Ly*np.around(Xee[:,:,1]/Ly)

		cee = np.sum(Xee*Xee[0][None,:], axis=2).mean(axis=1)           #la media e' sulle molecole
		cee = cee/cee[0]                                        #normalizzo rispetto al valore iniziale

		return cee

	def RIJ(self,t):
		rcut = 2.5
		tree = KDTree(self.X[t],boxsize=[2*self.box[t,1],2*self.box[t,3],np.Inf] )
		pairs = tree.query_pairs(rcut,output_type='ndarray')

		RIJ = self.X[t,pairs[:,0],:] - self.X[t,pairs[:,1],:]
		RIJ = self.wrap( t, RIJ )
		boolBOND = np.abs(pairs[:,0]-pairs[:,1]).astype(int) == 1

		return boolBOND,pairs,RIJ


	def Fpar(self,boolBOND,Rij): 		#forza totale agente su una particella gli passo gia' l'array che vien fuori da self.RIJ()
		R2 = np.linalg.norm(Rij,axis=1)

		#Forza LJ
		FLJ = np.zeros(Rij.shape)
		FLJ[:,0] = - 4*self.epsilon*(-12*(self.sigma**12/R2**13) + 6*(self.sigma**6/R2**7) )*Rij[:,0]/R2
		FLJ[:,1] = - 4*self.epsilon*(-12*(self.sigma**12/R2**13) + 6*(self.sigma**6/R2**7) )*Rij[:,1]/R2
		FLJ[:,2] = - 4*self.epsilon*(-12*(self.sigma**12/R2**13) + 6*(self.sigma**6/R2**7) )*Rij[:,2]/R2

		#Forza bond
		Fbond = np.zeros(Rij.shape)
		Fbond[boolBOND,0] = -self.k*(R2[boolBOND] - self.l0)*Rij[boolBOND,0]/R2[boolBOND]
		Fbond[boolBOND,1] = -self.k*(R2[boolBOND] - self.l0)*Rij[boolBOND,1]/R2[boolBOND]
		Fbond[boolBOND,2] = -self.k*(R2[boolBOND] - self.l0)*Rij[boolBOND,2]/R2[boolBOND]

		return FLJ + Fbond

	#*************************************************************************************************** 

def phi_integ(Rc):
	R = ReferenceFrame('R')
	Rmod = smp.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
	s,a,b = smp.symbols('s a b')
	vx,vy,vz = smp.symbols('v_x v_y v_z')
	ux,uy,uz = smp.symbols('u_x u_y u_z')
	V = vx*R.x + vy*R.y + vz*R.z
	U = ux*R.x + uy*R.y + uz*R.z
	A = V + s*U
	A2 = smp.sqrt(A & A)

	phi = (231/(128*smp.pi*Rc**3))*(1- (A2/Rc)**4)**2

	phi_int = 0
	for i in range(9):
		phi_int = phi_int +  (s**(9-i)/(9-i))*phi.diff(s,(8-i)).subs(s,0)/(smp.factorial(8-i))

	JJ = phi_int.subs(s,b) - phi_int.subs(s,a)
	JJf = smp.lambdify([vx,vy,vz,ux,uy,uz,a,b],JJ,'numpy')

	return JJf



def grid_node(g,rg,XX,rij,fij,Pairs,BOX):
	Rc = 2
	r = np.zeros( (Pairs.shape[0],3)) + rg[None,:]
	RRI = r - XX
	RRI[:,0] = RRI[:,0] - 2*BOX[1]*np.around( RRI[:,0]/(2*BOX[1]) )
	RRI[:,1] = RRI[:,1] - 2*BOX[3]*np.around( RRI[:,1]/(2*BOX[3]) )


	Rij = np.copy(rij)
	Fij = np.copy(fij)

	bolRC =  - np.sum(RRI*Rij,axis=1)**2/np.linalg.norm(Rij)**2  + np.linalg.norm(RRI,axis=1)**2 < Rc    	#tutte le coppie che sono nel supporto di phi
	if np.all(np.invert(bolRC)) :
		stress_node = np.zeros((3,3))
	else :
		RRI = RRI[bolRC]
		Rij = Rij[bolRC]
		Fij = Fij[bolRC]

		Smax = - np.sum(Rij*RRI,axis=1)/np.linalg.norm(Rij,axis=1)**2  + np.sqrt(  np.sum(RRI*Rij,axis=1)**2/np.linalg.norm(Rij)**2  - np.linalg.norm(RRI,axis=1)**2 +Rc**2 )
		Smin = - np.sum(Rij*RRI,axis=1)/np.linalg.norm(Rij,axis=1)**2  - np.sqrt(  np.sum(RRI*Rij,axis=1)**2/np.linalg.norm(Rij)**2  - np.linalg.norm(RRI,axis=1)**2 +Rc**2 )

		PHI_int = phi_integ(Rc) 									#funzione integrale di phi
		Integral  = PHI_int(RRI[:,0],RRI[:,1],RRI[:,2],Rij[:,0],Rij[:,1],Rij[:,2],Smin,Smax)[:,None]

		stress_node = - np.einsum('ab,ac->abc',Fij,Rij*Integral).sum(axis=0)      #+ np.eye(3)*np.sum( Ui*phi(r[None,:]-X,Rc) )   non divido per 2 perche' conto ogni coppia una sola volta

	return g, [stress_node[0,0],stress_node[1,1],stress_node[2,2],stress_node[0,1],stress_node[0,2],stress_node[1,2]]



def d2(t,Xt,Xt_dt,boxt,boxt_dt,Npar):
    tree = spatial.cKDTree(Xt,boxsize= [2*boxt[1],2*boxt[3],np.Inf])
    neigh = tree.query_ball_tree(tree,r=2.5)
    
    maxlen = [len(neigh[a]) for a in range(Npar) ]
    maxlen = max(maxlen)
    
    RIJt = np.zeros((Npar,maxlen,3))
    RIJt_dt = np.zeros((Npar,maxlen,3))
    
    for a in range(Npar):
        RIJt[a,:len(neigh[a]),:] =  - Xt[a][None,:] + Xt[neigh[a]]
        RIJt_dt[a,:len(neigh[a]),:] = - Xt_dt[a][None,:] + Xt_dt[neigh[a]]

    RIJt = wrap(RIJt , 2*boxt[1] , 2*boxt[3])
    RIJt_dt = wrap(RIJt_dt , 2*boxt_dt[1] , 2*boxt_dt[3])
    
    Y  = np.einsum('njk,njl->nkl' , RIJt_dt , RIJt_dt)
    X  = np.einsum('njk,njl->nkl' , RIJt , RIJt_dt)

    In = np.repeat( np.eye(3),Npar).reshape(3,3,Npar).T
    
    eps_ij = np.einsum( 'nik,njk -> nij' ,X,np.linalg.inv(Y)  ) - In
    W = np.einsum('njk,njl->nkl' , RIJt , RIJt)
    
    temp = np.einsum('njk,nij->nki',Y,In+eps_ij)
    
    D2 = np.einsum('nii',W) - 2*np.einsum('nij,nij->n',X,In+eps_ij) + np.einsum('nki,nik',temp,In+eps_ij)
    
    return t,D2
    


def main():
#	path = f'/home/gianfranco/unipi/film_trimeri/periodic/box_200.12.10/equilibratura/run1/lmp_data/Conf_*.bin'
	defo = np.arange(1,10)*20
	path = []
	for j in defo:
		path.append( f'/ddn/leporini/gianfranco/periodic/box_200.12.10/deform/run1/lmp_data/Conf_{j:d}.bin')

	print('import configurations')
	start = timer()
	particles = Particles(path,10,3,1,1,1111,0.9)		#inizializzazione della classe particelle a tutti i tempi
	end = timer()
	print(f'elapsed time: {end - start}')

	for j in range(len(defo)):
		step = j
	#	defo = 1 											#deformazione percentuale
	#	step = int((defo/50)*(particles.step.shape[0]-1))

		#calcolo a tempo fissato ###########################################################################################################
		print('constructing pairs')
		start = timer()

		boolBOND,pairs,RIJ = particles.RIJ(step)
		FIJ = particles.Fpar(boolBOND,RIJ)

		end = timer()
		print(f'force calculation time: {end - start}')

		X = particles.X[step,pairs[:,0],:]
		box = particles.box[step]
		grid = np.array( np.meshgrid( np.linspace(0,2*box[1],int(2*2*box[1])), np.linspace(0,2*box[3],int(2*2*box[3]) ), np.linspace(0,box[5]-box[4],2*int(box[5]-box[4])) ) )
		grid = grid.reshape(3,-1).T
		print(grid.shape)
		###############################################################################################
		#CALCOLO DELLO STRESS CG SU INTERA GRIGLIA
		values = np.arange(grid.shape[0])
		values = [ [g,grid[g],X,RIJ,FIJ,pairs,box] for g in range(grid.shape[0]) ]

		print('grid point evaluation')
		start1 = timer()
		###
		with Pool(80) as pool:
			G, stress = zip(*pool.starmap(grid_node,values))
		stress = np.array(stress)
		G = np.array(stress)
		GRID = grid(G)
		print('stress shape',stress.shape)
		np.savetxt(f'stress_CG_{defo[j]}.dat',np.c_[GRID,stress])
		###
		end1 = timer()
		print(f'total grid evaluation time: {end1 - start1}')

if __name__ == '__main__':
	main()












# Calcolo lo stress in punto della griglia
# def grid_node(X,Ui,box,grid,g,Rc,rcut,sigma,epsilon):
# 	LxMin = box[1]
# 	LxMax = box[2]
# 	Ly = box[3]
# 	LzMax = box[5]
# 	LzMin = box[4]

# 	r = grid[g]

# 	tree = spatial.KDTree(X,boxsize=[0,Ly,0])
# 	neigh = tree.query_ball_point(r, Rc)
# 	stress_node = np.zeros((3,3))


# 	if len(neigh) >0 :
# 		temp_tree = spatial.KDTree( X[neigh,:]  ,boxsize=[0,Ly,0])
# 		neigh_i = temp_tree.query_ball_tree(tree,rcut)

# 		#elimino la coppia ii
# 		for i, ni in enumerate(neigh):
# 			neigh_i[i].remove(ni)

# 		rij = np.ones([len(neigh_i),len(max(neigh_i,key = lambda x: len(x))),3])
# 		rri = np.zeros([len(neigh_i),len(max(neigh_i,key = lambda x: len(x))),3])
# 		lista_05 = np.zeros([len(neigh_i),len(max(neigh_i,key = lambda x: len(x)))])
# 		for i,j in enumerate(neigh_i):
# 			rij[i][0:len(j)] = X[neigh[i],:] - X[neigh_i[i],:]
# 			rri[i][0:len(j)] = r - X[neigh[i],:]  + X[neigh_i[i],:]*0
# 			lista_05[i][0:len(j)] = np.array([ 0.5 if (np.linalg.norm(X[neigh_i[i][k],:] - r)<Rc ) else 1 for k in range(len(j))])

# 		rij = rij.reshape((rij.shape[0]*rij.shape[1],3))
# 		rri = rri.reshape((rri.shape[0]*rri.shape[1],3))
# 		lista_05 = lista_05.ravel()[:,None]

# 		fij = flj(rij,sigma,epsilon)

# 		Smin, Smax = phi_interval(rri,rij,Rc)
# 		Int_func = phi_integ(Rc)
# 		integral = Int_func(rri[:,0],rri[:,1],rri[:,2],rij[:,0],rij[:,1],rij[:,2], Smin,Smax)[:,None]

# 		stress_node = - np.einsum('ab,ac->abc',fij,rij*lista_05*integral).sum(axis=0)   #+ np.eye(3)*np.sum( Ui*phi(r[None,:]-X,Rc) )

# 	return r, stress_node






















#Energia muro

# def Uwall(r,sigma,epsilon,rcut):
# 
	# d = smp.symbols('d')
# 
	# Uwallcut = - epsilon*((2/15)*(sigma/rcut)**9 - (sigma/rcut)**3 )
# 
	# Uwall93 = smp.Piecewise( (  epsilon*((2/15)*(sigma/d)**9 - (sigma/d)**3 ) + Uwallcut, d<=rcut ) , (0 , d> rcut)   )
# 
	# Uwall93 = smp.lambdify(d,Uwall93,'numpy')
# 
	# return Uwall93(np.abs(r))




# def createK(box,kval):
# 
    # L= np.c_[ box[:,1], box[:,2], box[:,4]-box[:,3] ].mean(axis=0)
# 
    # dk=np.pi/L
# 
    # tol=(0.5*dk).max()
# 
    # nbar=(np.ones(3)*kval/dk).astype(int)
# 
    # alln=[]
# 
    # for nx in range(-nbar[0],nbar[0]):
# 
        # for ny in range(-nbar[1],nbar[1]):
# 
            # for nz in range(-nbar[2],nbar[2]):
# 
                # alln.append((nx,ny,nz))
# 
    # alln=np.array(alln)
# 
    # allk=alln*dk
# 
    # mod=np.linalg.norm(allk,axis=1)
# 
    # allk=allk[np.argwhere((mod>(kval-tol))&(mod<kval+tol))].reshape(-1,3)
# 
    # return np.array(allk)
# 
# 
# 
# def self_intermediate_scattering_function(r,box,kval):
# 
    # nc=r.shape[0]
# 
    # npa=r.shape[1]
# 
    # isf=[]
# 
    # allk=createK(box,kval)
# 
    # for conf in range(1,nc):
# 
        # displ=r[conf]-r[0]
# 
        # isf.append(np.mean(np.cos(np.dot(allk,np.transpose(displ)))))
# 
    # return np.array(isf)
# 
# 



#calocolo energia interazione pareti per particella

# def Uwall_i(X,box,rcut,sigma,epsilon):
# 
	# Ui = np.zeros(X.shape[0])
# 
	# str1 = X[:,0]- box[1] < rcut
# 
	# Ui[str1] = Uwall( X[str1,0] - box[1],sigma,epsilon,rcut)
# 
# 
# 
	# str2 = box[2] - X[:,0] < rcut
# 
	# Ui[str2] = Uwall( X[str2,0] - box[2], sigma,epsilon,rcut )
# 
# 
# 
	# str3 = X[:,2] - box[4] < rcut
# 
	# Ui[str3] = Uwall( X[str3,2] - box[4],sigma,epsilon,rcut)
# 
# 
# 
	# return Ui
# 















# def main():
#	isf = self_intermediate_scattering_function(particles.data,particles.box,7.1)

	# for defo in [0,5,10,20,30,40,50]:
		# step = int((defo/50)*(particles.step.shape-1))
# 
		# data = particles.data[step]
		# box = particles.box[step]
# 
		# griglia
		# grid = np.array(np.meshgrid(np.linspace(box[1],box[2],int(2*box[2])),np.linspace(0,int(box[3]),int(box[3])),np.linspace(box[4],box[5],int((box[5]-box[4]))) ))
		# grid = grid.reshape(3,-1).T
# 
		# calcolo stress su tutta la griglia
		# start = timer()
# 
		# Ui = Uwall_i(data,box,rcut,sigma,epsilon)
# 
		# values = [ [data,Ui,box,grid,g,Rc,rcut,sigma,epsilon] for g in range(grid.shape[0]) ]
	# 
		# r = []
		# stress = []
		# for j in range(grid.shape[0]):
			# r.append(res[j][0])
			# stress.append(res[j][1])
# 
		# stress = np.array(stress)
		# r = np.array(r)
# 
		# np.savetxt(f'stress_CG_{defo:d}%.dat',np.c_[r[:,0],r[:,1],r[:,2],stress[:,0,0],stress[:,1,1],stress[:,2,2],stress[:,0,1],stress[:,0,2],stress[:,1,2]], header='rx \t ry \t rz \t sigmaXX \t sigmaYY \t sigmaZZ \t sigmaXY \t sigmaXZ \t sigmaYZ \t :  griglia con lato 1')
		# end = timer()
		# print(f'elapsed time: {end - start}')
