import numpy as np
from PP.pauli import *
import scipy.sparse as ss


L = 1000
A = np.diag(np.ones(L),-1)
SpA = ss.csr_matrix(A)
Ssx = ss.csr_matrix(sx)


print(np.kron(sx,A))


def kron_sx(a):
    sh = A.shape[0]
    new = np.zeros((2*sh,2*sh))
    new[:sh,sh:] = a
    new[sh:,:sh] = a
    return new

def kron_bmat(a):
    sh = A.shape
    return np.bmat([[np.zeros(sh), a],[a,np.zeros(sh)]])

