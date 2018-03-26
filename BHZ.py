import sys
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from PP.pauli import * #import pauli matrices s0,sx,sy,sz

u = -1.2
cs = 0
if len(sys.argv) == 3:
    u = float(sys.argv[1])
    cs = float(sys.argv[2])
N=10
s=500


c=cs*sy#coupling

k_range = np.linspace(-np.pi,np.pi,s)
di = np.eye(N)
od = np.diag(np.ones(N-1),1)

spectrum=np.zeros((N*4,s))

def buildH(k):
    H = np.kron(s0, np.kron(od, sz/2)) + np.kron(sz, np.kron(od, 1j*sx/2))
    H += H.H
    H += np.kron(s0, np.kron(di, (u + np.cos(k))*sz) + np.kron(di, np.sin(k)*sy))
    H += np.kron(sx, np.kron(di, c))
    return H

for i in np.arange(s):
    k = k_range[i]
    H=buildH(k) 
    vals, vecs = np.linalg.eigh(H)
    spectrum[:,i]=vals

plt.plot(k_range,spectrum.T)
plt.xlabel('$k_y$')
plt.ylabel('$E$')
plt.title('BHZ model')
plt.show()
