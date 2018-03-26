import numpy as np
import matplotlib.pyplot as plt
from PP.pauli import *
import sys

t1 = 1
t2 = 1j*t1/4
#m = 5*t1;
ms = np.linspace(1.5,0,100) 

plt.figure(figsize=(6, 3), facecolor='w')
plt.ion()
for a in np.arange(len(ms)):
    m=ms[a]
    if len(sys.argv) >= 2:
        #t2 = 1j*float(sys.argv[1])
        m  = float(sys.argv[1])

    N = 20
    s = 100
    k_range = np.linspace(-np.pi,np.pi,s)
    spectrum = np.zeros((s,2*N))
    def buildH(k,t1,t2,m):
        di = np.eye(N)
        du = np.diag(np.ones(N-1),1)
        dl = np.diag(np.ones(N-1),-1)
        
        #nn hopping
        H = -t1*np.kron(su, di) + 0*1j
        H += -t1*np.kron(su, dl)
        H += -t1*np.kron(su, di)*np.exp(-1j*k)
        
        #nnn-hopping
        H += t2*np.kron(sz,du)
        H += -t2*np.kron(sz,di)*np.exp(1j*k)
        H += -t2*np.kron(sz,du)*np.exp(-1j*k)

        #and backwards...
        H += H.H

        #mass terms
        H += m*np.kron(sz,di)
        return H

    for i in np.arange(s):
        H = buildH(k_range[i],t1,t2,m)
        vals, vecs = np.linalg.eigh(H)
        spectrum[i,:] = vals
    #plt.subplot(1,len(ms),a+1)
    plt.gcf().clear()
    plt.plot(spectrum)
    plt.xlabel('$k_y$')
    plt.title('m='+str(ms[a]))
    if a == 0:
        plt.ylabel('$E$')
    plt.tight_layout()
    #plt.show(block=False)
    plt.pause(0.03)
