#!/usr/bin/python2.7

import sys
import numpy as np

files=sys.argv[1:-1]
N=len(files)
outpath=sys.argv[-1]

data=[]

for f in files:
    data.append(np.loadtxt(f))
    
data = np.array(data)

cov_matrix = np.cov(data[:,:,1],rowvar=False)		#stimatore della matrice di covarianza cov(P_i,P_j). Per ottenere quella della media cov(P'_i,P'_j) = cov(P_i,P_j)/N

Potenza = np.mean(data[:,:,1],axis=0)
freq = data[0,:,0]

#calcolo del K_max+++++++++++++++++++++++++++++++++++++++++++++++++++
K_max = []

Lbox = data[0,0,-1]		#box
Leff = np.mean(data[:,1,-1])		#sistema
M = Potenza.max()
F = np.where(Potenza == M)[0][0]
K = freq[F]

KK = []
MM = []
dK = []

#lobo a sinistra del  picco***********************
ff = freq[:F]
pp = Potenza[:F]

i=1
while( i< len(ff)+1 ):
	if( pp[-i] >= M/2):
		KK.append([ ff[-i] ])
		MM.append([ pp[-i] ])
		if(i==len(ff)):
			break
	else:
		i=i-1
		break
	i= i+1


#lobo a destra del picco****************************
ff = freq[F:]		#contiene il massimo
pp = Potenza[F:]

j=0
while(j < len(ff) ):
	if( pp[j] >= M/2):
		KK.append([ ff[j] ])
		MM.append([ pp[j] ])
		if(j==len(ff)-1):
			break
	else:
		j=j-1
		break
	j = j+1
#***************************************************
	
KK = np.array(KK)
MM = np.array(MM) 

KK = np.sum(KK*MM)/np.sum(MM)	

I = F - i
J = F + j + 1

Area = np.sum(Potenza[I:J])
KP = np.sum(Potenza[I:J]*freq[I:J])


dK = (1.0/Area**4)*np.dot( np.dot(freq[I:J]*Area -KP,cov_matrix[I:J,I:J]),  freq[I:J]*Area -KP)  
dK = np.sqrt(dK)


K_max.append([Lbox,Leff,K,KK,dK])
K_max= np.array(K_max)   


f=open(outpath,'at')
np.savetxt(f,K_max,newline=' ')
f.write("\n")
