import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

N=100
s=100

w=1
spectrum=np.zeros((N,s))
for t in np.arange(0,s):
    u=np.sin(t/s*2*np.pi)
    v=1.2+np.cos(t/s*2*np.pi)

    a = np.empty((N,),float)
    a[::2] = 0
    a[1::2] = 1
    b = np.empty((N,),float)
    b[::2] = 1
    b[1::2] = 0
    offdiag=w*a[0:-1]+v*b[0:-1]
    diagonals=[-u*b+u*a,offdiag,offdiag]
    H=diags(diagonals,[0,-1,1])
    vals, vecs = np.linalg.eigh(H.toarray())
    spectrum[:,t]=vals
print("123")

plt.plot(spectrum.T)
plt.show()
