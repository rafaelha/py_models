import numpy as np
import PP.bosehubbard as bh
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm
import sys

L = 6 #number of sites
U = 1
t = 1

nl = [4,5,6,7,8]

plt.figure()

na = len(sys.argv)
if na >= 3:
    L = int(sys.argv[2])
elif na >= 2:
    U = float(sys.argv[1])

dt = 0.002 #time step
s = 900 #number of steps
times = np.arange(s)*dt;

for k in np.arange(len(nl)):
    # The model parameters
    L = nl[k]
    omegas = [0]*L
    links = [[i, (i+1) % L, t] for i in range(L)]
    links[-1] = [L-1, 0, 0] #open boundary conditions
    # Construct the model
    m = bh.Model(omegas, links, U)

    # Look at the single-particle hopping hamiltonian
    print(m.hopping)

    # Investigate the model with L bosons in it.
    m6 = m.numbersector(L)

    # Construct the many-body Hamiltonian (in sparse format)
    H = m6.hamiltonian
    basis = m6.basis

    #propagator
    Up = expm(-1j*dt*H)

    psi0 = np.array([1]*L) #quench from one boson per site
    psi0_i = basis.index(psi0)

    v0 = np.zeros(basis.size(L,L))
    v = np.zeros(basis.size(L,L))

    v0[psi0_i] = 1

    lohschmidt = np.zeros(s)

    for i in np.arange(s):
        v = Up.dot(v0)
        v0 = v
        lohschmidt[i] = np.abs(v[psi0_i])

    plt.plot(times,-1/L*np.log(np.absolute(lohschmidt)),label='L='+str(L))
    plt.legend()
    plt.xlabel('Time $t$')
    plt.ylabel('$g(t) = - \log(G(t))/N$')
plt.show(block=False)


