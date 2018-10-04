import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

N=70


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
"""
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

"""


steps=60
w=np.ones(steps)
v=np.linspace(2,0,steps)
spectrum=np.zeros((steps,N)).T
pos=np.zeros((steps,N)).T


#dist = np.repeat(np.abs( np.arange(N/2)), 2)
dist = np.repeat(np.abs( np.abs(np.arange(N/2)-(N/2-1)/2)), 2)

for i in np.arange(steps):
    diag[::2]=v[i];
    diag[1::2]=w[i];
    H=diags([diag,diag],[-1,1])
    vals, vecs = np.linalg.eigh(H.toarray())
    spectrum[:, i]=vals


    p = (np.abs(vecs.T)**2).dot(dist)
    pos[:, i] = p






nn = spectrum.shape[0]
plt.close('all')
plt.ion()    

fig, ax = plt.subplots(1,1, num="one")
fig.set_size_inches(3,3.7)

norm = plt.Normalize(np.min(pos), np.max(pos))

for i in np.arange(nn):
    y = spectrum[i,:] 
    
    points = np.array([v, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cdata = pos[i,:-1]

    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(cdata)
    lc.set_linewidth(1.5)
    line = ax.add_collection(lc)


#cbar = fig.colorbar(line, ax=ax)
#cbar.set_label('Expectation distance from center')
#plt.savefig('new_code_bandstructure.pdf')

plt.ylim((-3,3))
plt.xlim((0,2))


plt.xlabel('$v$')
plt.title('$w=1$')
plt.ylabel('Energy $E$')
plt.tight_layout()
