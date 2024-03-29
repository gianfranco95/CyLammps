cimport cython
import numpy as np
from scipy.spatial import KDTree
from libc.math cimport floor as FLOOR
from libc.math cimport abs as ABS
from libc.math cimport sqrt as SQRT
from libc.math cimport round as ROUND



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def initialize_bin(file):
	"""load lammps binary configuration	"""

	cdef double[:,:] X
	cdef double[:] box
	cdef Py_ssize_t N, Nfield, Nchunk, i

	N = np.fromfile(file,dtype=np.dtype('i8'),offset=8)[0]
	box = np.fromfile(file,dtype=np.dtype('6f8'), count =1,offset=44)[0]
	Nfield = np.fromfile(file,dtype=np.dtype('i4'), count =1,offset=92)[0]
	Nchunk = np.fromfile(file,dtype=np.dtype('i4'), count =1,offset=96)[0]

	Xnp = np.fromfile(file,dtype=np.dtype(f'i4,{int(Nfield*N/Nchunk)}f8'), count = Nchunk,offset=100)
	Xnp = np.array([Xnp[i][1].reshape(int(N/Nchunk),Nfield) for i in range(Nchunk)])
	X = Xnp.reshape(Xnp.shape[0]*Xnp.shape[1],Xnp.shape[2] )

	for i in range(X.shape[0]):
		X[i,0] =  X[i,0] - 2*box[1]*ROUND(X[i,0]/(2*box[1])) - box[0]
		X[i,1] =  X[i,1] - 2*box[3]*ROUND(X[i,1]/(2*box[3])) - box[2]
		X[i,2] =  X[i,2] - box[4]

	return N, X.base, box.base




@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def average_theta(double[:,:] R, double[:,:] Stress, double[:,:] C2t, double[:,:] C4t, double[:,:] S2t, double[:,:] S4t, Py_ssize_t p):
	""" compute the first harmonics of the coarse-grained stress"""

	cdef Py_ssize_t i,j,rr,M,N, Mx,My
	cdef double[:] X2,X4,Y2,Y4, count
	cdef double rc = 4

	Mx = <int>(Stress.shape[0]/2)
	My = <int>(Stress.shape[1]/2)

	if Stress.shape[0] < Stress.shape[1] :
		N = Stress.shape[0]
	else :
		N = Stress.shape[1]

	M = <int>(N/2)

	X2 = np.zeros(M)
	X4 = np.zeros(M)
	Y2 = np.zeros(M)
	Y4 = np.zeros(M)
	
	count = np.zeros(M)

	for i in range(-M,M):
		for j in range(-M,M):
			rr = <int>( FLOOR(R[M+ i,M + j]) )
			if rr < M:
				count[rr] += 1
				X2[rr] +=  Stress[Mx + j,My + i] * C2t[M + i,M + j]
				X4[rr] +=  Stress[Mx + j,My + i] * C4t[M + i,M + j]
				Y2[rr] +=  Stress[Mx + j,My + i] * S2t[M + i,M + j]
				Y4[rr] +=  Stress[Mx + j,My+ i] * S4t[M + i,M + j]

	for i in range(M):
		if (count[i] != 0):
			X2[i] = X2[i]/count[i]
			X4[i] = X4[i]/count[i]
			Y2[i] = Y2[i]/count[i]
			Y4[i] = Y4[i]/count[i]
		
		X2[i] = X2[i]*(i/rc)**p
		X4[i] = X4[i]*(i/rc)**p
		Y2[i] = Y2[i]*(i/rc)**p
		Y4[i] = Y4[i]*(i/rc)**p


	return X2.base, X4.base, Y2.base, Y4.base



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def CORR_STRESS(Py_ssize_t t):
	"""compute the stress autocorrelation matrix """

	cdef Py_ssize_t i, j, k, n, m, II, JJ, Nx, Ny, Nz, Mx, My						
	cdef double dz = 0.5		
	cdef Py_ssize_t Dg =2

	#differential stress
	P0 = 0*np.load(f'stress/stress_{t-1:d}.npy')
	P2 = np.load(f'stress/stress_{t:d}.npy')

	if (P0.shape[2] < P2.shape[2]) :
		P1 = np.zeros(P2.shape)
		P1[:,:,:P0.shape[2],:] = P0[:,:,:,:]
		del P0
	else :
		P1 = P0
		del P0

	cdef double[:,:,:,:] stress3d = P2 - P1
	del P1,P2


	cdef double[:] stress_medio							
	cdef double[:,:,:] stress

	Nx = stress3d.shape[0]
	Ny = stress3d.shape[1]
	Nz = stress3d.shape[2]

	Mx = <int>( FLOOR( min(Nx/2,Ny/2) )  )			
	My = Mx

	#integrate the stress along z 
	stress = np.zeros((Nx,Ny,3))
	stress_medio = np.zeros(3)

	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				stress[i,j,0] = stress[i,j,0] - dz*(stress3d[i,j,k,0] + stress3d[i,j,k,1])/2
				stress[i,j,1] = stress[i,j,1] + dz*(stress3d[i,j,k,0] - stress3d[i,j,k,1])/2
				stress[i,j,2] = stress[i,j,2] + dz*stress3d[i,j,k,3]

			stress_medio[0] = stress_medio[0] + stress[i,j,0]/(Nx*Ny)
			stress_medio[1] = stress_medio[1] + stress[i,j,1]/(Nx*Ny)
			stress_medio[2] = stress_medio[2] + stress[i,j,2]/(Nx*Ny)
	del stress3d
	#_______________________________________________

	cdef double[:,:,:] C = np.zeros((2*Mx,2*My,9))

	for n in range(-Mx,Mx):
		for m in range(-My,My):
			for i in range(Nx):
				if i+n >= Nx :
					II = i + n - Nx
				elif i+n <0 :
					II = Nx + i+n
				else:
					II = i+n

				for j in range(Ny):
					if j+m >= Ny :
						JJ = j + m - Ny
					elif j+m <0 :
						JJ = Ny + j+m
					else:
						JJ = j+m

					C[n+Mx,m+My,0] = C[n+Mx,m+My,0] + stress[i,j,0]*stress[II,JJ,0]/(Nx*Ny)
					C[n+Mx,m+My,1] = C[n+Mx,m+My,1] + stress[i,j,0]*stress[II,JJ,1]/(Nx*Ny)
					C[n+Mx,m+My,2] = C[n+Mx,m+My,2] + stress[i,j,0]*stress[II,JJ,2]/(Nx*Ny)
					C[n+Mx,m+My,3] = C[n+Mx,m+My,3] + stress[i,j,1]*stress[II,JJ,0]/(Nx*Ny)
					C[n+Mx,m+My,4] = C[n+Mx,m+My,4] + stress[i,j,1]*stress[II,JJ,1]/(Nx*Ny)
					C[n+Mx,m+My,5] = C[n+Mx,m+My,5] + stress[i,j,1]*stress[II,JJ,2]/(Nx*Ny)
					C[n+Mx,m+My,6] = C[n+Mx,m+My,6] + stress[i,j,2]*stress[II,JJ,0]/(Nx*Ny)
					C[n+Mx,m+My,7] = C[n+Mx,m+My,7] + stress[i,j,2]*stress[II,JJ,1]/(Nx*Ny)
					C[n+Mx,m+My,8] = C[n+Mx,m+My,8] + stress[i,j,2]*stress[II,JJ,2]/(Nx*Ny)

			C[n+Mx,m+My,0] = C[n+Mx,m+My,0] - stress_medio[0]*stress_medio[0]
			C[n+Mx,m+My,1] = C[n+Mx,m+My,1] - stress_medio[0]*stress_medio[1]
			C[n+Mx,m+My,2] = C[n+Mx,m+My,2] - stress_medio[0]*stress_medio[2]
			C[n+Mx,m+My,3] = C[n+Mx,m+My,3] - stress_medio[1]*stress_medio[0]
			C[n+Mx,m+My,4] = C[n+Mx,m+My,4] - stress_medio[1]*stress_medio[1]
			C[n+Mx,m+My,5] = C[n+Mx,m+My,5] - stress_medio[1]*stress_medio[2]
			C[n+Mx,m+My,6] = C[n+Mx,m+My,6] - stress_medio[2]*stress_medio[0]
			C[n+Mx,m+My,7] = C[n+Mx,m+My,7] - stress_medio[2]*stress_medio[1]
			C[n+Mx,m+My,8] = C[n+Mx,m+My,8] - stress_medio[2]*stress_medio[2]

	np.save(f'Correlation_stress/corr_{t:d}.npy' , C.base, allow_pickle=False,fix_imports=False)
	return


@cython.cdivision(True)
@cython.boundscheck(False)
def LOG_THERMO(Py_ssize_t  t):
	"""compute thermodynamics of the total system"""
	cdef double epsilon = 1
	cdef double sigma = 1
	cdef double Km=1111
	cdef double l0 = 0.9
	cdef Py_ssize_t Rc=2
	cdef double rcut=2.5

	file = f'../lmp_data/Conf_{t:d}.bin'
	cdef double[:,:] X
	cdef double[:] box
	cdef Py_ssize_t N
	cdef double Lx

	N,X,box = initialize_bin(file)
	Lx = box[1] - box[0]

	cdef Py_ssize_t i, j, k,  Nx, Ny, Nz, p, Np
	cdef double Dg = 2
	cdef double VOLUME =0

	Nx = <int>( FLOOR(box[1]*2*Dg))
	Ny = <int>( FLOOR(box[3]*2*Dg))
	Nz = <int>( FLOOR((box[5]-box[4])*Dg))

	tree2D = KDTree(X.base[:,:2],boxsize=[2*box[1],2*box[3]])
	tree2Dgrid = KDTree( np.array( np.meshgrid( np.arange(Nx)/Dg, np.arange(Ny)/Dg ,indexing='ij') ).reshape(2,-1).T  , boxsize=[2*box[1],2*box[3]])
	lista = tree2Dgrid.query_ball_tree( tree2D, Rc )

	for i in range(Nx):
		for j in range(Ny):
			VOLUME +=  np.sort( X.base[lista[i*Ny +j],2], kind='mergesort')[-3:].mean() * (1/Dg**2)
				
	del tree2D
	del tree2Dgrid
	del lista

	cdef double RIJx,RIJy,RIJz,FIJx,FIJy,FIJz

	cdef Py_ssize_t[:,:] pairs
	tree = KDTree(X,boxsize=[2*box[1],2*box[3],np.Inf] )
	pairs = ( tree.query_pairs(rcut,output_type='ndarray')).astype(np.int_)
	Np = pairs.shape[0]

	cdef double[:] THERMO = np.zeros( 9 )	

	THERMO[0] = Lx
	THERMO[8] = VOLUME
	
	for p in range(Np):
		RIJx = -X[pairs[p,0],0] + X[pairs[p,1],0]
		RIJy = -X[pairs[p,0],1] + X[pairs[p,1],1]
		RIJz = -X[pairs[p,0],2] + X[pairs[p,1],2]
		RIJx =  RIJx - 2*box[1]*ROUND(RIJx/(2.0*box[1]))
		RIJy =  RIJy - 2*box[3]*ROUND(RIJy/(2.0*box[3]))
		R = SQRT( RIJx**2 +  RIJy**2 + RIJz**2 )

		if (  ( <int>(ABS(FLOOR(<double>pairs[p,0] /3.0) - FLOOR(<double>pairs[p,1] /3.0) )) == 0 )  &  ( <int>(ABS(pairs[p,0] - pairs[p,1]) ) == 1 )  ):
			FIJx =  Km*(R - l0)*RIJx/R
			FIJy =  Km*(R - l0)*RIJy/R
			FIJz =  Km*(R - l0)*RIJz/R
			THERMO[7] +=  (0.5 *Km* (R - l0)**2)/N
		else:
			FIJx =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJx/R
			FIJy =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJy/R
			FIJz =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJz/R
			THERMO[7] +=   (4.0*epsilon*( (sigma**12/R**12) - (sigma**6/R**6) )  - 4.0*epsilon*( (sigma**12/rcut**12) - (sigma**6/rcut**6) ) )/N

		THERMO[1] +=   (FIJx * RIJx)/(VOLUME)
		THERMO[2] +=   (FIJy * RIJy)/(VOLUME)
		THERMO[3] +=   (FIJz * RIJz)/(VOLUME)
		THERMO[4] +=   (FIJx * RIJy)/(VOLUME)
		THERMO[5] +=   (FIJx * RIJz)/(VOLUME)
		THERMO[6] +=   (FIJy * RIJz)/(VOLUME)


	cdef double Fw

	for n in range(N):
		if (X[n,2] <= rcut) :
			Fw = epsilon * ( -6*sigma**9/(5*X[n,2]**10) + 3*sigma**3/X[n,2]**4 )
			THERMO[3] +=  Fw*(-X[n,2])/(VOLUME)
			THERMO[7] += (epsilon *( 2*sigma**9/(15*X[n,2]**9) - sigma**3/X[n,2]**3 )   - epsilon *( 2*sigma**9/(15*rcut**9) - sigma**3/rcut**3 ))/N

	return THERMO.base


@cython.cdivision(True)
@cython.boundscheck(False)
def CGSTRESS(Py_ssize_t  t , Py_ssize_t Dg = 2):
	"""compute the coarse grained stress"""

	cdef double epsilon = 1
	cdef double sigma = 1
	cdef double Km=1111
	cdef double l0 = 0.9
	cdef Py_ssize_t Rc=2
	cdef double rcut=2.5

	file = f'../lmp_data/Conf_{t:d}.bin'
	cdef double[:,:] X
	cdef double[:] box
	cdef Py_ssize_t N

	N,X,box = initialize_bin(file)

	cdef Py_ssize_t i, j, k, Np, p, Nx, Ny, Nz

	Nx = <int>( FLOOR(box[1]*2*Dg))
	Ny = <int>( FLOOR(box[3]*2*Dg)) 
	Nz = <int>( FLOOR((box[5]-box[4])*Dg))


	cdef double[:,:] PROFILO = np.zeros( (Nx,Ny) )

	tree2D = KDTree(X.base[:,:2],boxsize=[2*box[1],2*box[3]])
	tree2Dgrid = KDTree( np.array( np.meshgrid( np.arange(Nx)/Dg, np.arange(Ny)/Dg ,indexing='ij') ).reshape(2,-1).T  , boxsize=[2*box[1],2*box[3]])
	lista = tree2Dgrid.query_ball_tree( tree2D, 2 )

	for i in range(Nx):
		for j in range(Ny):
			PROFILO[i,j] = np.sort( X.base[lista[i*Ny +j],2], kind='mergesort')[-3:].mean() 
			
	del tree2D
	del tree2Dgrid
	del lista
	
	cdef double RIJx,RIJy,RIJz,FIJx,FIJy,FIJz,rrix,rriy,rriz,rrjx,rrjy,rrjz,R,u2,uv,v2,w2,H,xj,xi,a,b, phi_int
	cdef double pi = np.pi

	cdef Py_ssize_t[:,:] pairs
	tree = KDTree(X,boxsize=[2*box[1],2*box[3],np.Inf] )
	pairs = ( tree.query_pairs(rcut,output_type='ndarray')).astype(np.int_)
	Np = pairs.shape[0]

	cdef double[:,:,:,:] PRESS = np.zeros( (  Nx, Ny, Nz, 6 )  )
	cdef Py_ssize_t xc, yc, zc, Dx, Dy, Dz, II, JJ, KK

	for p in range(Np):
		RIJx = -X[pairs[p,0],0] + X[pairs[p,1],0]
		RIJy = -X[pairs[p,0],1] + X[pairs[p,1],1]
		RIJz = -X[pairs[p,0],2] + X[pairs[p,1],2]
		RIJx =  RIJx - 2*box[1]*ROUND(RIJx/(2.0*box[1]))
		RIJy =  RIJy - 2*box[3]*ROUND(RIJy/(2.0*box[3]))
		R = SQRT( RIJx**2 +  RIJy**2 + RIJz**2 )

		xc = <int>( Dg*(X[pairs[p,0],0] + RIJx/2.0) )
		yc = <int>( Dg*(X[pairs[p,0],1] + RIJy/2.0) )
		zc = <int>( Dg*(X[pairs[p,0],2] + RIJz/2.0) )

		Dx = <int>( ABS(RIJx)*Dg/2.0 +Rc*Dg +1 )
		Dy = <int>( ABS(RIJy)*Dg/2.0 +Rc*Dg +1 )
		Dz = <int>( ABS(RIJz)*Dg/2.0 +Rc*Dg +1 )


		if (  ( <int>(ABS(FLOOR(<double>pairs[p,0] /3.0) - FLOOR(<double>pairs[p,1] /3.0) )) == 0 )  &  ( <int>(ABS(pairs[p,0] - pairs[p,1]) ) == 1 )  ):
			FIJx =  Km*(R - l0)*RIJx/R
			FIJy =  Km*(R - l0)*RIJy/R
			FIJz =  Km*(R - l0)*RIJz/R
		else:
			FIJx =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJx/R
			FIJy =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJy/R
			FIJz =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJz/R

		for i in range(xc-Dx, xc+Dx+1):
			if i>= Nx :
				II = i - Nx 
			elif i<0 :
				II = Nx + i
			else:
				II = i
			for j in range(yc-Dy, yc+Dy+1):
				if j>= Ny :
					JJ = j - Ny 
				elif j<0 :
					JJ = Ny + j
				else:
					JJ = j
				for k in range(zc-Dz, zc+Dz+1):
					if ( (k< Nz) & (k>=0) & ( <double>k/<double>Dg < PROFILO[II,JJ] ) ) :
						KK = k
						rrix = - (<double>II)/(<double>Dg) + X[pairs[p,0],0]
						rriy = - (<double>JJ)/(<double>Dg) + X[pairs[p,0],1]
						rriz = - (<double>KK)/(<double>Dg) + X[pairs[p,0],2]
						rrix =  rrix - 2*box[1]*ROUND(rrix/(2.0*box[1]))
						rriy =  rriy - 2*box[3]*ROUND(rriy/(2.0*box[3]))

						rrjx = - (<double>II)/(<double>Dg) + X[pairs[p,1],0]
						rrjy = - (<double>JJ)/(<double>Dg) + X[pairs[p,1],1]
						rrjz = - (<double>KK)/(<double>Dg) + X[pairs[p,1],2]
						rrjx =  rrjx - 2*box[1]*ROUND(rrjx/(2.0*box[1]))
						rrjy =  rrjy - 2*box[3]*ROUND(rrjy/(2.0*box[3]))

						u2 = R**2
						v2 = rrix**2 + rriy**2 + rriz**2
						w2 = rrjx**2 + rrjy**2 + rrjz**2
						uv = RIJx*(rrix)  + RIJy*(rriy) + RIJz*(rriz)
						H = SQRT((- uv**2 + u2*v2)/u2)
						xj = (RIJx*(rrjx)  + RIJy*(rrjy) + RIJz*(rrjz))/R
						xi = uv/R

						if H < Rc :
							if ( (v2<=Rc**2) & (w2<=Rc**2) ) :
								a = xi
								b = xj
							elif ( (v2<=Rc**2) & (w2>Rc**2)  ):
								a = xi
								b =  SQRT(Rc**2 - H**2)*xj/ABS(xj)
							elif ( (v2>Rc**2) & (w2<=Rc**2)  ):
								a =  SQRT(Rc**2 - H**2)*xi/ABS(xi)
								b =  xj
							elif ( (v2>Rc**2) & (w2>Rc**2) & (xi*xj < 0) ):
								a = SQRT(Rc**2 - H**2)*xi/ABS(xi)
								b = SQRT(Rc**2 - H**2)*xj/ABS(xj)
							else :
								a=0
								b=0

							phi_int = b + (b/Rc**8)*(b**8/9.0  + 4.0*b**6*H**2/7.0  + 6.0*b**4*H**4/5.0 + 4.0*b**2*H**6/3.0 + H**8)  - 2.0*(b/Rc**4)*( b**4/5.0 + 2.0*b**2*H**2/3.0  + H**4)
							phi_int = phi_int -a  -(a/Rc**8)*(a**8/9.0  + 4.0*a**6*H**2/7.0  + 6.0*a**4*H**4/5.0 + 4.0*a**2*H**6/3.0 + H**8)  + 2.0*(a/Rc**4)*( a**4/5.0 + 2.0*a**2*H**2/3.0  + H**4)
							phi_int = ABS( phi_int*231.0/(128.0*pi*Rc**3*R) )

							PRESS[II,JJ,KK,0] =  PRESS[II,JJ,KK,0] + FIJx*RIJx*phi_int
							PRESS[II,JJ,KK,1] =  PRESS[II,JJ,KK,1] + FIJy*RIJy*phi_int
							PRESS[II,JJ,KK,2] =  PRESS[II,JJ,KK,2] + FIJz*RIJz*phi_int
							PRESS[II,JJ,KK,3] =  PRESS[II,JJ,KK,3] + FIJx*RIJy*phi_int
							PRESS[II,JJ,KK,4] =  PRESS[II,JJ,KK,4] + FIJx*RIJz*phi_int
							PRESS[II,JJ,KK,5] =  PRESS[II,JJ,KK,5] + FIJy*RIJz*phi_int
						


	#wall interaction
	cdef double rrwx,rrwy,rrwz, zi,zw ,Fw, Rw,Ri
	cdef Py_ssize_t n

	for n in range(N):
		if (X[n,2] <= rcut) :
			xc = <int>( Dg*X[n,0] )
			yc = <int>( Dg*X[n,1] )
			zc = <int>( Dg*X[n,2] )

			Dx = <int>(Rc*Dg +1 )
			Dy = <int>(Rc*Dg +1 )
			Dz = <int>(Rc*Dg +1 )

			Fw = epsilon * ( -6*sigma**9/(5*X[n,2]**10) + 3*sigma**3/X[n,2]**4 )

			for i in range(xc-Dx, xc+Dx+1):
				if i>= Nx :
					II = i - Nx
				elif i<0 :
					II = Nx + i
				else:
					II = i
				for j in range(yc-Dy, yc+Dy+1):
					if j>= Ny :
						JJ = j - Ny
					elif j<0 :
						JJ = Ny + j
					else:
						JJ = j
					for k in range(0, zc+Dz+1):
						if ( <double>k/<double>Dg < PROFILO[II,JJ]  ) :
							KK = k
							rrix = - (<double>II)/(<double>Dg) + X[n,0]
							rriy = - (<double>JJ)/(<double>Dg) + X[n,1]
							rriz = - (<double>KK)/(<double>Dg) + X[n,2]
							rrix =  rrix - 2*box[1]*ROUND(rrix/(2.0*box[1]))
							rriy =  rriy - 2*box[3]*ROUND(rriy/(2.0*box[3]))

							H  =  SQRT( rrix**2  + rriy**2 )

							rrwx = rrix
							rrwy = rriy
							rrwz = - (<double>KK)/(<double>Dg) + 0.0

							Rw = SQRT(rrwx**2 + rrwy**2 + rrwz**2)
							Ri = SQRT(rrix**2 + rriy**2 + rriz**2)

							zi = rriz
							zw = rrwz

							if H < Rc :
								if ( (Ri<=Rc) & (Rw<=Rc) ) :
									a = zi
									b = zw
								elif ( (Ri<=Rc) & (Rw>Rc)  ):
									a = zi
									b = SQRT(Rc**2 - H**2)*zw/ABS(zw)
								elif ( (Ri>Rc) & (Rw<=Rc)  ):
									a =  SQRT(Rc**2 - H**2)*zi/ABS(zi)
									b =  zw
								elif ( (Ri>Rc) & (Rw>Rc) & (zi*zw < 0) ):
									a = SQRT(Rc**2 - H**2)*zi/ABS(zi)
									b = SQRT(Rc**2 - H**2)*zw/ABS(zw)
								else :
									a=0
									b=0

								phi_int = b + (b/Rc**8)*(b**8/9.0  + 4.0*b**6*H**2/7.0  + 6.0*b**4*H**4/5.0 + 4.0*b**2*H**6/3.0 + H**8)  - 2.0*(b/Rc**4)*( b**4/5.0 + 2.0*b**2*H**2/3.0  + H**4)
								phi_int = phi_int -a  -(a/Rc**8)*(a**8/9.0  + 4.0*a**6*H**2/7.0  + 6.0*a**4*H**4/5.0 + 4.0*a**2*H**6/3.0 + H**8)  + 2.0*(a/Rc**4)*( a**4/5.0 + 2.0*a**2*H**2/3.0  + H**4)
								phi_int = ABS( phi_int*231.0/(128.0*pi*Rc**3*ABS(X[n,2])) )

								PRESS[II,JJ,KK,2] =  PRESS[II,JJ,KK,2] + Fw*(-X[n,2])*phi_int


	np.save(f'stress/stress_{t:d}.npy' , PRESS.base, allow_pickle=False,fix_imports=False)

	return




@cython.cdivision(True)
@cython.boundscheck(False)
def PROFILO(Py_ssize_t t, Py_ssize_t Dg=2):
	"""compute topografy of the free surface"""
	cdef double[:,:] X
	cdef double[:] box
	cdef Py_ssize_t i, j, k, N, Nx, Ny, Nz

	N,X,box = initialize_bin(f'../lmp_data/Conf_{t:d}.bin')
	Nx = <int>( FLOOR(box[1]*2*Dg))
	Ny = <int>( FLOOR(box[3]*2*Dg))
	Nz = <int>( FLOOR((box[5]-box[4])*Dg))	

	cdef double[:,:] PROFILO = np.zeros( (Nx,Ny) )

	tree2D = KDTree(X.base[:,:2],boxsize=[2*box[1],2*box[3]])
	tree2Dgrid = KDTree( np.array( np.meshgrid( np.arange(Nx)/Dg, np.arange(Ny)/Dg ,indexing='ij') ).reshape(2,-1).T  , boxsize=[2*box[1],2*box[3]])
	lista = tree2Dgrid.query_ball_tree( tree2D, 2 )

	for i in range(Nx):
		for j in range(Ny):
			PROFILO[i,j] = np.sort( X.base[lista[i*Ny +j],2], kind='mergesort')[-3:].mean()


	np.save(f'profilo/profilo_{t:d}.npy' , PROFILO.base, allow_pickle=False,fix_imports=False)

	return 


@cython.cdivision(True)
@cython.boundscheck(False)
def MAPPA_DH(Py_ssize_t t,Py_ssize_t t2, Py_ssize_t Nx):
	"""compute the height variation as a function of the strain and the system length """
	cdef double[:,:] X, X2, DHxy
	cdef double[:] box, box2, DH
	cdef Py_ssize_t i, j, N, Ny, Nz
	cdef Py_ssize_t Dg = 2
	cdef double Lx
	file = f'../lmp_data/Conf_{t:d}.bin'
	file2 = f'../lmp_data/Conf_{t2:d}.bin'

	N,X2,box2 = initialize_bin(file2)

	N,X,box = initialize_bin(file)
	Lx = 2*box[1]

	Ny = <int>( FLOOR(box[3]*2*Dg))
	Nz = <int>( FLOOR((box[5]-box[4])*Dg))

	DHxy = np.zeros( (Nx,Ny) )
	DH = np.zeros(Nx)

	tree2D = KDTree(X.base[:,:2],boxsize=[2*box[1],2*box[3]])
	tree2Dgrid = KDTree( np.array( np.meshgrid( np.arange(Nx)*Lx/Nx , np.arange(Ny)/Dg ,indexing='ij') ).reshape(2,-1).T  , boxsize=[2*box[1],2*box[3]])
	lista = tree2Dgrid.query_ball_tree( tree2D, 2 )
	
	for i in range(Nx):
		for j in range(Ny):
			ids = np.argsort( X.base[lista[i*Ny +j],2], kind='mergesort')[-3:]
			DHxy[i,j] = (X2.base[lista[i*Ny +j],2][ids] - X.base[lista[i*Ny +j],2][ids]).mean()

	for i in range(Nx):
		for j in range(Ny):
			DH[i] = DH[i] + DHxy[i,j]/Ny
	return t2, DH.base



@cython.boundscheck(False) 
@cython.wraparound(False)  
@cython.cdivision(True)
def VOLUME(Py_ssize_t t):
	"""compute the real volume"""
	cdef double Lx,Ly,dx,dy,m, Vol
	cdef double[:] box, H
	cdef double[:,:] X,Profilo
	cdef Py_ssize_t N, Nx, Ny, i, j, a ,b

	N,X,box  = initialize_bin(f'../lmp_data/Conf_{t:d}.bin')

	Lx = 2*box[1]
	Ly = 2*box[3]

	Nx = <int>( FLOOR(Lx) )
	Ny = <int>( FLOOR(Ly) )
	dx = Lx/Nx
	dy = Ly/Ny
	Profilo = np.zeros((Nx,Ny))

	for i in range(N):
		a = int(FLOOR((X[i,0])/dx))
		b = int(FLOOR((X[i,1])/dy))
		if(a<Nx and b<Ny):
			m = Profilo[a,b]
			if (X[i,2]>m):
				Profilo[a,b] = X[i,2]

	Vol = 0  
	for i in range(Nx):
		for j in range(Ny):
			Vol = Vol + Profilo[i,j]*dx*dy


	return t, Vol
