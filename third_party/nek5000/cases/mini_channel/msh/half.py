import numpy as np 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--Ny',default=8,type=int)
parser.add_argument('--Ly',default=1.0,type=float)
parser.add_argument('--beta',default=1.0,type=float)
args=parser.parse_args()

betay = args.beta
Ly    = args.Ly
Ny    = args.Ny
yspace = np.linspace(0,Ly,Ny)
ptdst = np.linspace(0.0,Ly,Ny)
ptdst = 0.5*(np.tanh(betay*(2.0*ptdst-1.0))/np.tanh(betay)+1.0)
print(ptdst)