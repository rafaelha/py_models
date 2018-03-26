import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

N=20


vv=0
ww=1

diag=np.zeros(N-1)
diag[::2]=vv
diag[1::2]=ww
H=diags([diag,diag],[-1,1])
vals, vecs = np.linalg.eigh(H.toarray())

kpoints=np.linspace(-np.pi,np.pi,10)
wpoints=np.linspace(-2,2,10)
eta=2


def green(k,w,sl):
    sub=np.zeros(N)
    sub[sl::2]=np.arange(-N/4,N/4)
    cr=np.exp(1j*sub*k)/np.sqrt(N/2)
    return -np.imag(np.sum(np.absolute(vecs.T.dot(cr))**2/(w-(vals-np.min(vals))-1j*eta)))

k = np.linspace(0-np.pi, np.pi, 100)
w = np.linspace(-2, 12, 100)

result=np.zeros((w.size,k.size))
result1=np.zeros((w.size,k.size))
result2=np.zeros((w.size,k.size))

for x in np.arange(k.size):
    for y in np.arange(w.size):
        result1[y,x]=green(k[x],w[y],0)
        result2[y,x]=green(k[x],w[y],1)

plt.subplot(131)
plt.imshow(result1+result2)
plt.subplot(132)
plt.imshow(result1)
plt.subplot(133)
plt.imshow(result2)
plt.show(block=False)




steps=100
w=np.ones(steps)
v=np.linspace(2,0,steps)
spectrum=np.zeros((steps,N))

for i in np.arange(steps):
    diag[::2]=v[i];
    diag[1::2]=w[i];
    H=diags([diag,diag],[-1,1])
    vals, vecs = np.linalg.eigh(H.toarray())
    spectrum[i,:]=vals
plt.figure()            
plt.plot(v,spectrum)
plt.axvline(x=vv)
plt.show()
