cimport cython
import numpy as np
from scipy.spatial import KDTree
from libc.math cimport floor as FLOOR
from libc.math cimport abs as ABS
from libc.math cimport sqrt as SQRT
from libc.math cimport round as ROUND


#************************************************************************************************************************************$
#------------------------------------------------------------------------------------------------------------------------------------$
#************************************************************************************************************************************$
#CALCOLO DEL PRODOTTO SCALARE DELLO STRESS CON SENI E COSENI
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def ESH_DECOMPOSITION(Py_ssize_t t):
	cdef double[:,:,:] Corr = np.load(f'Correlation_stress/corr_{t:d}.npy')
#	cdef double[:,:,:] Corr = np.load(f'Correlation_stress/diff_corr_{t:d}.npy')

	cdef double dz = 0.5
	cdef Py_ssize_t Nx,Mx
	Nx = Corr.shape[0]   #Ny = Nx
	Mx = <int>(Nx/2)

	cdef double[:,:] X,Y,R,Ct,St,C2t,S2t,C4t,S4t
	x = np.arange(0,Nx) - (Mx-0.5)
	X, Y = np.meshgrid(x,x)
	del x
	R = np.sqrt(X.base**2 + Y.base**2)
	Ct = X.base/R.base
	St = Y.base/R.base
	C2t = Ct.base**2 - St.base**2
	S2t = 2*Ct.base*St.base
	C4t = C2t.base**2 - S2t.base**2
	S4t = 2*C2t.base*S2t.base

	cdef double[:,:] X_2, X_4, Y_2, Y_4    #le componenti sono in ordine 11,22,33,12,13,23
	cdef double pi = np.pi
	cdef double rc = 2/dz			#sarebbe il raggio di coars-grain in unita' della griglia

	X_2 = np.zeros((Mx,6))
	X_4 = np.zeros((Mx,6))
	Y_2 = np.zeros((Mx,6))
	Y_4 = np.zeros((Mx,6))

	cdef double C33_0 = (Corr[Mx,Mx,8] + Corr[Mx-1,Mx,8] + Corr[Mx-1,Mx-1,8] + Corr[Mx,Mx-1,8])/4	#correlaione 33 calcolate in r=0
	cdef Py_ssize_t i,j,rr,val,k

	for i in range(Nx):
		for j in range(Nx):
			rr = <int>( FLOOR(R[i,j]) )
			if rr < Mx:
				for k,val in enumerate([0,4,8,1,2,5]):
					X_2[rr,k] += -(rr/rc)**2*( Corr[j,i,val]/C33_0 )*C2t[i,j]/(2*pi**2)
					X_4[rr,k] += -(rr/rc)**2*( Corr[j,i,val]/C33_0 )*C4t[i,j]/(2*pi**2)
					Y_2[rr,k] += -(rr/rc)**2*( Corr[j,i,val]/C33_0 )*S2t[i,j]/(2*pi**2)
					Y_4[rr,k] += -(rr/rc)**2*( Corr[j,i,val]/C33_0 )*S4t[i,j]/(2*pi**2)

	np.save(f'Correlation_stress/X_2_{t:d}.npy' , X_2.base, allow_pickle=False,fix_imports=False)
	np.save(f'Correlation_stress/X_4_{t:d}.npy' , X_4.base, allow_pickle=False,fix_imports=False)
	np.save(f'Correlation_stress/Y_2_{t:d}.npy' , Y_2.base, allow_pickle=False,fix_imports=False)
	np.save(f'Correlation_stress/Y_4_{t:d}.npy' , Y_4.base, allow_pickle=False,fix_imports=False)

#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
#CALCOLO DELLA MATRICE DI AUTOCORRELAZIONE DELLO STRESS

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def CORR_STRESS(Py_ssize_t t):
	cdef Py_ssize_t i, j, k, n, m, II, JJ, Nx, Ny, Nz, Mx, My						#dimensioni della griglia su cui e' calcolato lo stress 2d(integrato su z)
	cdef double dz = 0.5										#step della griglia 3d su cui e' calcolato lo stress coarse-grained
	cdef Py_ssize_t Dg =2

	#stress differenziale________________________________________________________________
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

	#stress attuale________________________________________________________________________
#	cdef double[:,:,:,:] stress3d = np.load(f'stress/stress_{t:d}.npy')
	#___________________________________________________________________________________________________________

	cdef double[:] stress_medio							#stress 3 componenti mediato sulla griglia 2d
	cdef double[:,:,:] stress

	Nx = stress3d.shape[0]
	Ny = stress3d.shape[1]
	Nz = stress3d.shape[2]

	#Mx = <int>( FLOOR( min(Nx/2,Ny/2)*np.cos(np.pi/4) )  )			#sono sicuro di non sommare contributi di immagine periodica
	Mx = <int>( FLOOR( min(Nx/2,Ny/2) )  )			#sono sicuro di non sommare contributi di immagine periodica
	My = Mx

	#integro lo stress lungo z __________________
	stress = np.zeros((Nx,Ny,3))						#stress2d componenti sferiche s1,s2,s3 calcolata sulla base cartesiana
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

	cdef double[:,:,:] C = np.zeros((2*Mx,2*My,9))			#matrice di autocorrelazione dello stress 2d

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

#	np.save(f'Correlation_stress/diff_corr_{t:d}.npy' , C.base, allow_pickle=False,fix_imports=False)
	np.save(f'Correlation_stress/corr_{t:d}.npy' , C.base, allow_pickle=False,fix_imports=False)
	return

#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
# FUNZIONE PER IL CALCOLO DELLO STRESS E DELLA TOPOGRAFIA SUPERFICIALE

@cython.cdivision(True)
@cython.boundscheck(False)
def CGSTRESS(Py_ssize_t  t , Py_ssize_t Dg = 2):
	#PARAMETRI ####################################################################
	cdef double epsilon = 1
	cdef double sigma = 1
	cdef double Km=1111
	cdef double l0 = 0.9
	cdef Py_ssize_t Rc=2
	cdef double rcut=2.5
	###############################################################################
	#LETTURA CONFIGURAZIONE AL 'TEMPO' t
	file = f'../lmp_data/Conf_{t:d}.bin'
	cdef double[:,:] X
	cdef double[:] box
	cdef Py_ssize_t N

	N,X,box = initialize_bin(file)

	######################################################################################################################################
	#CREAZIONE GRIGLIA
	cdef Py_ssize_t i, j, k, Np, p, Nx, Ny, Nz

	Nx = <int>( FLOOR(box[1]*2*Dg))
	Ny = <int>( FLOOR(box[3]*2*Dg)) 
	Nz = <int>( FLOOR((box[5]-box[4])*Dg))


	#CALCOLO DELLA TOPOGRAFIA ##############################################################################################################
	#uso bin circolare raggio 2
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
	
	#CALCOLCO DELLO STRESS COARSE-GRAINED ##################################################################################################
	cdef double RIJx,RIJy,RIJz,FIJx,FIJy,FIJz,rrix,rriy,rriz,rrjx,rrjy,rrjz,R,u2,uv,v2,w2,H,xj,xi,a,b, phi_int
	cdef double pi = np.pi

	cdef Py_ssize_t[:,:] pairs
	tree = KDTree(X,boxsize=[2*box[1],2*box[3],np.Inf] )
	pairs = ( tree.query_pairs(rcut,output_type='ndarray')).astype(np.int_)
	Np = pairs.shape[0]

	cdef double[:,:,:,:] PRESS = np.zeros( (  Nx, Ny, Nz, 6 )  )
	cdef Py_ssize_t xc, yc, zc, Dx, Dy, Dz, II, JJ, KK

	#termine di interazione a coppia
	for p in range(Np):
		#vettore  della coppia
		RIJx = -X[pairs[p,0],0] + X[pairs[p,1],0]
		RIJy = -X[pairs[p,0],1] + X[pairs[p,1],1]
		RIJz = -X[pairs[p,0],2] + X[pairs[p,1],2]
		RIJx =  RIJx - 2*box[1]*ROUND(RIJx/(2.0*box[1]))
		RIJy =  RIJy - 2*box[3]*ROUND(RIJy/(2.0*box[3]))
		R = SQRT( RIJx**2 +  RIJy**2 + RIJz**2 )

		#coordinate del nodo piu' vicino al centro di massa della coppia e Dx
		xc = <int>( Dg*(X[pairs[p,0],0] + RIJx/2.0) )
		yc = <int>( Dg*(X[pairs[p,0],1] + RIJy/2.0) )
		zc = <int>( Dg*(X[pairs[p,0],2] + RIJz/2.0) )

		Dx = <int>( ABS(RIJx)*Dg/2.0 +Rc*Dg +1 )
		Dy = <int>( ABS(RIJy)*Dg/2.0 +Rc*Dg +1 )
		Dz = <int>( ABS(RIJz)*Dg/2.0 +Rc*Dg +1 )


		#calcolo forze
		if (  ( <int>(ABS(FLOOR(<double>pairs[p,0] /3.0) - FLOOR(<double>pairs[p,1] /3.0) )) == 0 )  &  ( <int>(ABS(pairs[p,0] - pairs[p,1]) ) == 1 )  ):
			#print('BOND',pairs[i,0],pairs[i,1])
			FIJx =  Km*(R - l0)*RIJx/R
			FIJy =  Km*(R - l0)*RIJy/R
			FIJz =  Km*(R - l0)*RIJz/R
		else:
			#print('LJ',pairs[i,0],pairs[i,1])
			FIJx =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJx/R
			FIJy =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJy/R
			FIJz =  4.0*epsilon*(-12.0*(sigma**12/R**13) + 6.0*(sigma**6/R**7) )*RIJz/R

		#3x forLoop sulla griglia intorno la coppia
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
							#calcolo estremi dell'integrale phi
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
							#print('PHI',xi,xj,a,b,phi_int,FIJx,R,  FIJx*RIJx*phi_int)

							PRESS[II,JJ,KK,0] =  PRESS[II,JJ,KK,0] + FIJx*RIJx*phi_int
							PRESS[II,JJ,KK,1] =  PRESS[II,JJ,KK,1] + FIJy*RIJy*phi_int
							PRESS[II,JJ,KK,2] =  PRESS[II,JJ,KK,2] + FIJz*RIJz*phi_int
							PRESS[II,JJ,KK,3] =  PRESS[II,JJ,KK,3] + FIJx*RIJy*phi_int
							PRESS[II,JJ,KK,4] =  PRESS[II,JJ,KK,4] + FIJx*RIJz*phi_int
							PRESS[II,JJ,KK,5] =  PRESS[II,JJ,KK,5] + FIJy*RIJz*phi_int
						


	#CALCOLO INTERAZIONE MURO ###################################################################################################################
	cdef double rrwx,rrwy,rrwz, zi,zw ,Fw, Rw,Ri
	cdef Py_ssize_t n

	for n in range(N):
		if (X[n,2] <= rcut) :
			#coordinate i rimappate sulla griglia NxNyNz
			xc = <int>( Dg*X[n,0] )
			yc = <int>( Dg*X[n,1] )
			zc = <int>( Dg*X[n,2] )

			Dx = <int>(Rc*Dg +1 )
			Dy = <int>(Rc*Dg +1 )
			Dz = <int>(Rc*Dg +1 )

			#forza del muro diretta lungo z
			Fw = epsilon * ( 6*sigma**9/(5*X[n,2]**10) - 3*sigma**3/X[n,2]**4 )

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
								#calcolo estremi dell'integrale phi
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

								PRESS[II,JJ,KK,2] =  PRESS[II,JJ,KK,2] - Fw*X[n,2]*phi_int


	np.save(f'stress/stress_{t:d}.npy' , PRESS.base, allow_pickle=False,fix_imports=False)

	return




#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
#CALCOLO DELLA TOPOGRAFIA #####################################################################################################################################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
def PROFILO(Py_ssize_t t, Py_ssize_t Dg=2):
	cdef double[:,:] X

	cdef double[:] box
	cdef Py_ssize_t i, j, k, N, Nx, Ny, Nz

	#uso bin circolare raggio 2
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

	#CALCOLO DELLA DENSITA' #########################
	#in realta' calcolo il numero di pallette dentro una bolla di raggio 4, non divido per il volume della bolla, questo e' fatto a posteriori
#	tree = KDTree(X,boxsize=[2*box[1],2*box[3],np.Inf] )
#	cdef Py_ssize_t[:,:,:] DENSITY = np.zeros((Nx,Ny,Nz)).astype(np.int_)

#	for i in range(Nx):
#		for j in range(Ny):
#			for k in range(Nz):
#				if (<double>k/<double>Dg < PROFILO[i,j]) :
#					DENSITY[i,j,k] = DENSITY[i,j,k] + <int>(tree.count_neighbors(   KDTree(np.array([<double>i/<double>Dg, <double>j/<double>Dg, <double>k/<double>Dg]).reshape(1,3), boxsize=[2*box[1],2*box[3],np.Inf] ), 4 ))

	np.save(f'profilo/profilo_{t:d}.npy' , PROFILO.base, allow_pickle=False,fix_imports=False)
#	np.save(f'density/density_{t:d}.npy' , DENSITY.base, allow_pickle=False,fix_imports=False)

	return 


#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
#MAPPA DI H(x)  AL VARIARE DELLA COMPRESSIONE #####################################################################################################################################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
def MAPPA_H(Py_ssize_t t, Py_ssize_t Nx):
	#usare come Nx un valore doppio di  Lx, quella nominale 200,300,700 non quella proprio esatta
	cdef double[:,:] X, PROFILO
	cdef double[:] box, H
	cdef Py_ssize_t i, j, N, Ny, Nz
	cdef Py_ssize_t Dg = 2
	cdef double Lx
	file = f'../lmp_data/Conf_{t:d}.bin'

	N,X,box = initialize_bin(file)
	Lx = 2*box[1]

	Ny = <int>( FLOOR(box[3]*2*Dg))
	Nz = <int>( FLOOR((box[5]-box[4])*Dg))

	PROFILO = np.zeros( (Nx,Ny) )
	H = np.zeros(Nx)

	tree2D = KDTree(X.base[:,:2],boxsize=[2*box[1],2*box[3]])
	tree2Dgrid = KDTree( np.array( np.meshgrid( np.arange(Nx)*Lx/Nx , np.arange(Ny)/Dg ,indexing='ij') ).reshape(2,-1).T  , boxsize=[2*box[1],2*box[3]])
	lista = tree2Dgrid.query_ball_tree( tree2D, 2 )

	for i in range(Nx):
		for j in range(Ny):
			PROFILO[i,j] = np.sort( X.base[lista[i*Ny +j],2], kind='mergesort')[-3:].mean()

	for i in range(Nx):
		for j in range(Ny):
			H[i] = H[i] + PROFILO[i,j]/Ny

	return t, H.base



#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
#MAPPA DI DeltaH(x,eps)  AL VARIARE DELLA COMPRESSIONE #####################################################################################################################################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
def MAPPA_DH(Py_ssize_t t,Py_ssize_t t2, Py_ssize_t Nx):
	#usare come Nx un valore doppio di  Lx, quella nominale 200,300,700 non quella proprio esatta
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

#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
#CALCOLO DEL VOLUME ##################################################################
@cython.boundscheck(False) 
@cython.wraparound(False)  
@cython.cdivision(True)
def VOLUME(Py_ssize_t t):
	cdef double Lx,Ly,dx,dy,m, Vol
	cdef double[:] box, H
	cdef double[:,:] X,Profilo
	cdef Py_ssize_t N, Nx, Ny, i, j, a ,b

	N,X,box  = initialize_bin(f'../lmp_data/Conf_{t:d}.bin')

	Lx = 2*box[1]
	Ly = 2*box[3]

	#CREAZIONE PROFILO**********************************

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

	Vol = 0   #volume
	for i in range(Nx):
		for j in range(Ny):
			Vol = Vol + Profilo[i,j]*dx*dy


	return t, Vol



#*********************************************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#*********************************************************************************************************************************************************************************************
#importa configurazione binaria LAMMPS ###############################################################################################################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def initialize_bin(file):
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
