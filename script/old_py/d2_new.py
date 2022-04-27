from libreria import *



def d2_step(j,dj):
	path = []
	path.append( f'/ddn/leporini/gianfranco/periodic/box_200.12.10/deform/run1/lmp_data/Conf_{j:d}.bin')
	path.append( f'/ddn/leporini/gianfranco/periodic/box_200.12.10/deform/run1/lmp_data/Conf_{j-dj:d}.bin')

	X,box,step,boundary,Npar = read_lammps(path,2,False)
	wrap_coord(X,box)

	d2 = D2(X[1],X[0],box[1],box[0],Npar)

	return step[1],d2


def main():
	defo = np.arange(50000)

	print('d2 step')
	start = timer()
	res = d2_step(100,1)

	end = timer()
	print(f'elapsed time: {end - start}')
	print(res[1].max())
	print(res[1].min())


if __name__ == '__main__':
        main()
