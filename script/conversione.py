import numpy as np
import sys

CONF = int(sys.argv[1])


def initialize_bin(file):
    N = np.fromfile(file,dtype=np.dtype('i8'),offset=8)[0]
    box = np.fromfile(file,dtype=np.dtype('6f8'), count =1,offset=44)[0]
    Nfield = np.fromfile(file,dtype=np.dtype('i4'), count =1,offset=92)[0]
    Nchunk = np.fromfile(file,dtype=np.dtype('i4'), count =1,offset=96)[0]

    Xnp = np.fromfile(file,dtype=np.dtype(f'i4,{int(Nfield*N/Nchunk)}f8'), count = Nchunk,offset=100)
    Xnp = np.array([Xnp[i][1].reshape(int(N/Nchunk),Nfield) for i in range(Nchunk)])
    X = Xnp.reshape(Xnp.shape[0]*Xnp.shape[1],Xnp.shape[2] )

    return X, box


X,box  = initialize_bin(f'lmp_data/Conf_{CONF:d}.bin')
N = X.shape[0]

imx =  np.around(X[:,0]/(2*box[1])).astype(np.int_)
imy =  np.around(X[:,1]/(2*box[3])).astype(np.int_)
imz = np.zeros(N).astype(np.int_)
Id = np.arange(N)+1


f = open(f"start/Conf_{CONF:d}_start.dump", "w")

f.write("ITEM: TIMESTEP\n")
f.write("0\n")
f.write("ITEM: NUMBER OF ATOMS\n")
f.write(f"{N:d}\n")
f.write("ITEM: BOX BOUNDS pp pp fs\n")
f.write(f"{box[0]:.16f} {box[1]:.16f}\n")
f.write(f"{box[2]:.16f} {box[3]:.16f}\n")
f.write(f"{box[4]:.16f} {box[5]:.16f}\n")
f.write("ITEM: ATOMS id xu yu zu ix iy iz\n")

for i in range(N):
    f.write(f"{Id[i]:d} {X[i,0]:.16f} {X[i,1]:.16f} {X[i,2]:.16f} {imx[i]:d} {imy[i]:d} {imz[i]:d}\n")


f.close()

