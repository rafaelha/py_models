import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

N =10
s=100
u_range = np.linspace(-4,4,s)
spectrum = np.zeros((s,N*2))
def buildh(u):
    t = 1
    d = t
    hop = -t*np.ones(N-1)
    en=diags([-u*np.ones(N),hop,hop],[0,1,-1]).toarray()

    gap= d*np.ones(N-1)
    ph=diags([gap,-gap],[-1,1]).toarray()

    H=np.zeros((2*N,2*N))
    H[0:N,0:N]=en
    H[N:,N:]=-en
    H[0:N,N:]=ph
    H[N:,0:N] = ph
    return H
    
for t in np.arange(s):
    vals, vecs = np.linalg.eigh(buildh(u_range[t]))
    spectrum[t,:] = vals
plt.figure()
plt.plot(u_range,spectrum)
plt.xlabel('$\mu/t$')
plt.ylabel('$E$')
plt.show()
