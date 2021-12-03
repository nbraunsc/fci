import pyscf
import numpy as np
import sys
from pyscf import gto, scf, ao2mo, fci, cc

molecule = '''
H      0.00       0.00       0.00
H      0.00       0.00       1.00
H      0.00       0.00       2.00
H      0.00       0.00       3.00'''

charge = 0
spin  = 0
#basis_set = 'ccpvdz'
basis_set = '6-31g'

orb_basis = 'scf'
cas = True
cas_nstart = 1
cas_nstop = 10
cas_nel = 10

pmol = PyscfHelper()
#pmol.init(molecule,charge,spin,basis_set,orb_basis,cas,cas_nstart,cas_nstop,cas_nel)
pmol.init(molecule,charge,spin,basis_set,orb_basis)


#do some stuff to get integrals and save to disk
np.save('data/ints_0b.npy',pmol.ecore)
np.save('data/ints_1b.npy',pmol.h)
np.save('data/ints_2b.npy',pmol.g)
