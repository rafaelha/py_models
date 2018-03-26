import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

N=50
s=300

u=-1.5

#Define pauli matrices
sx = np.matrix([[0,1],[1,0]])
sy = np.matrix([[0,-1j],[1j,0]])
sz = np.matrix([[1,0],[0,-1]])

k_range = np.linspace(-np.pi,np.pi,s)

spectrum=np.zeros((N*2,s))
for i in np.arange(s):
    k = k_range[i]

    H = np.kron(0.5*(sz+1j*sx),np.diag(np.ones(N-1),-1))
    H += H.H
    H += np.kron(np.cos(k)*sz + np.sin(k)*sy + u*sz,np.eye(N))
    vals, vecs = np.linalg.eigh(H)
    spectrum[:,i]=vals

plt.plot(k_range,spectrum.T)
plt.show()
