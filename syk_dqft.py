#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:43:27 2017

@author: ocan
"""
from sys import getsizeof
import numpy as np
import numpy.matlib
import random
import time
import matplotlib.pyplot as plt
import scipy.linalg as la
n=24; ## MAGNETIC FLUX QUANTA - THIS IS DOUBLE THE NUMBER OF COMPLEX FERMIONS
size = int(n)
halfsize = int(n/2)
print('N = '+str(n))

perms=[]

J=1;
J_sigma=np.sqrt((6*J**2)/(n**3))

time0=time.clock()

for i in range(size):
    for j in range(i+1,size):
        for k in range(j+1,size):
            for l in range(k+1,size):
                perms.append((random.gauss(0,J_sigma),np.asarray([i,j,k,l])))    
time1=time.clock()

print ('PERMS were constructed in '+str(int(time1-time0)) + ' seconds')

def binary(i, n):
    return list(map(int,str(np.binary_repr(i,width=int(n/2))))) 

def unbinary(bin):
    my_lst_str = ''.join(map(str,bin))
    return np.int(my_lst_str,2)

def act(state,index):
    '''
    state: occupation number representation NOW THIS WORKS. 
    changes the occupation 1to0 or 0to1 and gives the sign from commutations.    
    '''
    prefac=1; #change this if maj_tilda is acting.
    if index >= int(n/2):
        prefac= (1-2*state[index % int(n/2)])*1.0j
    index = index % int(n/2)
    factor=(-1)**sum(state[0:index])*prefac #sign
    state[index]=1-state[index] #change occupation
    return factor,state


def matrix_element(state,perm):
    ''' 
        evaluates the action of the matrix element (perm) on the given (state)
        creation and annihilation works the same way (flip the occupation between 0 and 1)
        so we just need to figure out the overall sign.
    '''
    factor=perm[0] #start with the random J and keep updating with -1 signs from commutations.
    for ind in perm[1]: #act for each creation operator within the term.
        state=act(state[1],ind)
        coeff=state[0]
        factor=factor*coeff
    return factor,state[1]

time2=time.clock()

H=np.zeros((2**int(n/2),2**int(n/2)),dtype='c8'); #Create the zero matrix.

time_est_0=time.clock()
ct = 0
for m in range(int(2**(n/2))): #RUN OVER ALL STATES
    ''' Time: '''
    if m >= 0.0001*2**(n/2) and ct == 0:
        time_est_1=time.clock()
        ct = 1;
        print (str(((time_est_1-time_est_0)*1.0/(m*60))*(2**(n/2)))+ ' mins estimated')
    ''' --Time-- '''
    bin_store=binary(m,n)
    for perm in perms: #now, for a given state run over terms of the H - for each i,j,k,l    
        initial_state=bin_store[:]  
        factor,final_state=matrix_element((1,initial_state),perm)
        H[unbinary(final_state),m]+=factor #CHECK THE ORDER? M,N OR N,M?
time3=time.clock()
print ('matrix was constructed in '+str(int(time3-time2)) + ' seconds')

#now diagonalize
eigvals, eigvecs = la.eigh(H)

time4=time.clock()

'''
np.save('eigvals',eigvals)
np.save('eigvecs',eigvecs)
'''

'''

fig = plt.figure()
ax = fig.add_subplot(121)
cax=ax.matshow(H.real)
fig.colorbar(cax)
ax = fig.add_subplot(122)
cax2=ax.matshow(H.imag)
fig.colorbar(cax2)
plt.show()
'''
#eigvals, eigvecs = la.eigh(H_dima)

time4=time.clock()
#plt.plot(eigvals[0:10])
#np.save('eigvals',eigvals)
#np.save('eigvecs',eigvecs)

print ('matrix was diagonalized in '+str(int(time4-time3)) + ' seconds')


#now time evolution on some state
psi0=np.ones(eigvals.size)/np.sqrt(eigvals.size)
psi=np.array(psi0)

dt = 0.001
nt = 4000
times = np.arange(nt)*dt;
#propagator
U = la.expm(-1j*H*dt)
lohschmidt=np.zeros(nt)

for t in np.arange(nt):
    psi=U.dot(psi)
    lohschmidt[t]=np.abs(np.vdot(psi,psi0))**2

plt.figure()
#plt.subplot(121)
#plt.plot(lohschmidt)
#plt.xlabel('Time t')
#plt.ylabel('Lohschmidt echo')
#plt.subplot(122)
plt.plot(times,-np.log(lohschmidt))
plt.xlabel('Time t')
plt.ylabel('$g(t) = - \log(G(t))/N$')

plt.show(block=False)
    



