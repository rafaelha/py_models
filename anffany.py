import numpy as np
from numpy import sin, cos, kron
import matplotlib.pyplot as plt
from PP.pauli import *
import sys

a = 20 #lattice constant Angstrom
a2 = a**2 # material parameters: Cano et al. (Bernevig group) PRB 95 161306 (2017)
C0 = -0.0145
C1 = 10.59
C2 = 11.5
A = 0.889
M0 = 0.0205
M1 = -18.77
M2 = -13.5

delta = 0.01 # SC order parameter (real)

mu = 0 # chemical potential

Lx = 40 # number of layers in x-direction
Ly = 28
Lz = 100

N = Lx*Ly

ky = np.arange(-Ly//2,Ly//2)*np.pi/(Ly*a)
kz = np.arange(-Lz//2,Lz//2)*np.pi/(Lz*a)

def buildH1(y,z):
    hop = np.diag(np.ones(Lx - 1),1) # + np.diag(np.ones(Lx - 1),-1) # nn-hopping
    di = np.eye(Lx) # unit matrix
    eps = ( C0 + 2*C1/a2*(1 - cos(z*a)) + 2*C2/a2*(2 - cos(y*a)) ) * di \
           - C2/a2 * (hop + hop.T)
    M = ( M0 + 2*M1/a2*(1 - cos(z*a)) + 2*M2/a2*(2 - cos(y*a)) ) * di \
         - M2/a2 * (hop + hop.T)
    H0 = kron( kron(eps, s0), s0) \
         + kron( kron(M, sz) + A*(kron(1j/(2*a)*(hop-hop.T),sx) + kron(1/a*sin(y*a)*di,sy)) , sz)
    
    return H0

def buildH2(x,z):
    di = np.eye(4*Lx) # unit matrix
    H1 = buildH1(x,z)
    dt = np.zeros((Lx,Lx)) # diag matrix that that is only one for top- and bottom layer
    dt[0,0] = 1
    dt[Lx-1,Lx-1] = 1
    dtt = kron(kron(dt,s0),s0)
    H2 = kron(H1 - mu*di, sz) + kron(delta*dtt, sx)
    return H2

def n(x,y): # map 2d real space to 1d vector-indices
    return int(Lx*y + x) 

def buildHopping(hopping):
    H = np.zeros((Lx*Ly,Lx*Ly))
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            if hopping == 'x': #x-hopping (OBC)
                if i < Lx-1:
                    H[n(i,j),n(i+1,j)] = 1
               # elif i > 0:
                #    H[n(i,j),n(i-1,j)] = 1
            elif hopping == 'y': # y-hopping (PBC)
                H[n(i,j),n(i,(j+1)%Ly)] = 1
                #H[n(i,j),n(i,(j-1)%Ly)] = 1
    return np.asmatrix(H)

def buildDelta(phi):
    DD = np.zeros((Lx*Ly,Lx*Ly), dtype=np.dtype(complex))
    for i in [0,Lx-1]:
        for j in np.arange(3*Ly/4,Ly):
            DD[n(i,j),n(i,j)] = delta*np.exp(1j*phi)
        for j in np.arange(Ly/4):
            DD[n(i,j),n(i,j)] = delta
    return kron(kron(np.asmatrix(DD),s0),s0)

di = np.eye(N)
di4 = np.eye(4*N)
hx = buildHopping('x')
hx_plus = hx + hx.T
hx_minus = hx - hx.T

hy = buildHopping('y')
hy_plus = hy + hy.T
hy_minus = hy - hy.T

hop = hx_plus + hy_plus

def buildH3_once(phi):
    AA = A*(kron(1j/(2*a)*(hx_minus),sx) + kron(1j/(2*a)*(hy_minus),sy))
    Mdelta = buildDelta(phi)
    MD = kron(Mdelta.real, sx) + kron(Mdelta.imag, -sy)
    return MD, AA

def buildH3(z, MD, AA):
    eps = (C0 + 2*C1/a2*(1 - cos(z*a)) + 4*C2/a2) * di \
            - C2/a2*( hop )
    M = (M0 + 2*M1/a2*(1 - cos(z*a)) + 4*M2/a2) * di \
            - M2/a2*( hop )
    
    H0 = kron( kron(eps, s0), s0) + kron( kron(M, sz) + AA , sz) #minus sign -1j???
    
    H_BdG = kron(H0 - mu*di4, sz) + MD
    return H_BdG

def diag(ham='H1'):
    z = 1
    if ham == 'H2':
        z = 2 
    spectrum = np.zeros((Lz,Ly,z*4*Lx))
    for j in np.arange(len(kz)):
        for i in np.arange(len(ky)):
            if ham == 'H1':
                H = buildH1(ky[i],kz[j])
            elif ham == 'H2':
                H = buildH2(ky[i],kz[j])
            vals, vecs = np.linalg.eigh(H)
            spectrum[j,i,:] = vals
    spectrum = spectrum.reshape((Lz,z*4*Lx*Ly),order='C') # project everything on kz-axis 
    return spectrum

def plot(spectrum):
    plt.figure(figsize=(6,6), facecolor='w')
    plt.plot(kz*a, spectrum*1000,linewidth=0.1)
    plt.ylim((-20,20))
    plt.xlim((-np.pi/2,np.pi/2))        
    plt.xlabel('$k_z a$')
    plt.ylabel('$E$ [meV]')
    plt.tight_layout()
    plt.ion()

#s = diag('H2')
#plot(s)

def build():
    phi = np.pi
    MD, AA = buildH3_once(phi)
    H = buildH3(1, MD, AA)

def diag3():
    phi = np.pi
    spectrum = np.zeros((Lz,8*N))

    MD, AA = buildH3_once(phi)

    for j in np.arange(len(kz)):
        H = buildH3(kz[j], MD, AA)
        vals, vecs = np.linalg.eigh(H)
        spectrum[j,:] = vals
        print(j)

    plot(spectrum)

