import kwant

# For plotting
from matplotlib import pyplot


a=1 
t=1.0
W=10
L=30

# Start with an empty tight-binding system and a single square lattice.
# `a` is the lattice constant (by default set to 1 for simplicity.
lat = kwant.lattice.square(a)

syst = kwant.Builder()

syst[lat.neighbors()] = -t
syst = syst.finalized()

print(syst.hamiltonian_submatrix())
