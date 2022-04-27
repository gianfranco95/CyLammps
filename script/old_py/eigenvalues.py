#!/usr/bin/python3

import sys
import re
from timeit import default_timer as timer
import numpy as np
from multiprocessing import Pool,Process,Queue
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


def func(i):
	files=sys.argv[1]
	el=[[],[],[]]
	data=np.loadtxt(files,skiprows=4800*i,max_rows=4800)
	for l in range(len(data)):
		k=3*l
		if(np.all(data[l] != np.zeros(3)) ):
			el[0].append( k)
			el[0].append( k+1)
			el[0].append( k+2)
			el[1].append( i)
			el[1].append( i)
			el[1].append( i)
			el[2].append( data[l,0])
			el[2].append( data[l,1])
			el[2].append( data[l,2])
	return el


def main():
	N=4800
	val= list(range(0,1))

	start=timer()

	with Pool() as pool:
		res = pool.map(func,val)

	end=timer()
	print("tempo = {:f}".format(-start+end))	

	col=[]
	row=[]
	data=[]

	for r in res:
		col.append(r[0][0])
		col.append(r[0][1])
		col.append(r[0][2])
		row.append(r[1][0])
		row.append(r[1][1])
		row.append(r[1][2])
		data.append(r[2][0])
		data.append(r[2][1])
		data.append(r[2][2])

	
	print(col)

	M = csr_matrix( (data,(row,col)),shape=(3*N,3*N) )
	
	end1 = timer()
	print("fine = {:f}".format(end1-end))

	eigvl,eigvc = eigsh(M, 2000, which='LM')

	np.savetxt('eigenvalues.dat',eigvl)
	np.savetxt('eigenvectors.dat',eigvc)
	
	return


if __name__ == "__main__":
	main()
