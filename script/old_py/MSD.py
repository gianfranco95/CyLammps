#!/usr/bin/python2.7

import sys
import re
import numpy as np

#IMPORT DATA****************************************
outpath1 = sys.argv[-2]
outpath2 = sys.argv[-1]
files=sys.argv[2:-2]

L0=float(sys.argv[1])
Nt = len(files) 
#**************************************************
def timestep(string):
  return map(int,re.findall(r'\d+',string))[-1]

files.sort(key=timestep)
Nt = len(files)

def MSD1(j,Msd,X0,Msd_esp):
	data = np.array([np.loadtxt(files[j],usecols=(2,3,4),skiprows=9)])
	Lx =  2*np.loadtxt(files[j],usecols=(1),skiprows=5,max_rows=1)
	Ly =  2*np.loadtxt(files[j],usecols=(1),skiprows=6,max_rows=1)
	
	#*****************************************
	x= data[0,:,0]
	y= data[0,:,1] 
	yp = data[0,:,1] - Ly*np.around(data[0,:,1]/(Ly))
	z= data[0,:,2]

	Lx1 = x.max() - x.min()	

	if(j==0):
		X0.append([x,y,z,Lx])	
		Msd.append([Lx,0,0,0,0,Lx1])
		
	else:		
		if(j==50):
			X0.append([x,y,z,Lx])
			Msd_esp.append([Lx1,0,0,0,0,Lx])			
		msd_x1 = x - X0[0][0]		#affine +  non affine
		msd_x2 = x - (X0[0][0]/L0)*Lx	#non affine
		msd_y = y - X0[0][1]
		msd_z = z - X0[0][2]		


		Msd.append([Lx1,np.mean(msd_x1**2,axis=0),np.mean(msd_x2**2,axis=0),np.mean(msd_y**2,axis=0),np.mean(msd_z**2,axis=0),Lx])
		
		if(j>50):				#sola espansione
			msd_x1esp = (x - X0[1][0])
			msd_x2esp = x - (X0[1][0]/X0[1][3])*Lx
			msd_yesp = (y - X0[1][1])
			msd_zesp = (z - X0[1][2])	
			Msd_esp.append([Lx1,np.mean(msd_x1esp**2,axis=0),np.mean(msd_x2esp**2,axis=0),np.mean(msd_yesp**2,axis=0),np.mean(msd_zesp**2,axis=0),Lx]) 

	return;



def MSD2(j,Msd,X0,Msd_esp):
        data = np.array([np.loadtxt(files[j],usecols=(2,3,4),skiprows=9)])
        Lx =  2*np.loadtxt(files[j],usecols=(1),skiprows=5,max_rows=1)
        Ly =  2*np.loadtxt(files[j],usecols=(1),skiprows=6,max_rows=1)

        #CREAZIONE PROFILO**********************************
        x= data[0,:,0]
        y= data[0,:,1]
        yp = data[0,:,1] - Ly*np.around(data[0,:,1]/(Ly))
        z= data[0,:,2]

	Lx1 = x.max() - x.min()	

        if(j==0):
                X0.append([x,y,z,Lx])
        else:
             	if(j==1):
                        X0.append([x,y,z,Lx])
			Msd_esp.append([Lx1,0,0,0,0,Lx])
					
                msd_x1 = x - X0[0][0]           #estrinseco
                msd_x2 = x - (X0[0][0]/L0)*Lx
                msd_y = y - X0[0][1]
                msd_z = z - X0[0][2]

                Msd.append([Lx1,np.mean(msd_x1**2,axis=0),np.mean(msd_x2**2,axis=0),np.mean(msd_y**2,axis=0),np.mean(msd_z**2,axis=0),Lx])

                if(j>1):                               #sola espansione
                        msd_x1esp = (x - X0[1][0])
                        msd_x2esp = x - (X0[1][0]/X0[1][3])*Lx
                        msd_yesp = (y - X0[1][1])
                        msd_zesp = (z - X0[1][2])
                        Msd_esp.append([Lx1,np.mean(msd_x1esp**2,axis=0),np.mean(msd_x2esp**2,axis=0),np.mean(msd_yesp**2,axis=0),np.mean(msd_zesp**2,axis=0),Lx])

        return;




#Chiamata funzione
X0 = []
Msd = []
Msd_esp= []


for i in range(Nt):
	if(Nt==101):
		MSD1(i,Msd,X0,Msd_esp)
	else:
		MSD2(i,Msd,X0,Msd_esp)

Msd = np.array(Msd)
Msd_esp = np.array(Msd_esp)


np.savetxt(outpath1,Msd)
np.savetxt(outpath2,Msd_esp)

		
