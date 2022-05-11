from quspin.operators import hamiltonian,quantum_operator
from quspin.basis import spin_basis_1d
import numpy as np
#import matplotlib.pyplot as plt


def real_generate():
    "generate a disorder realization"
    np.random.seed()
    disorder = -1 + 2 * np.random.ranf((basis.L,))
    h_disorder_z = [[0.5*disorder[i],i] for i in range(basis.L)]
    h_disorder = [["x",h_disorder_z]]
    no_checks = {"check_herm":False,"check_pcon":False,"check_symm":False}
    H_dis = hamiltonian(h_disorder,[],basis=basis,dtype=np.float64,**no_checks)
    return H_dis


def J_zz_gen(J_max,a,L):
    '''
    generate all long-range interaction terms for lattice with L sites,
    with long-range exponent a, parameter J_max, and opening boundary conditions
    '''
    JJ = []
    for i in range(0,L):
        for j in range(i+1,L):
            JJ.append([J_max/(j-i)**a,i,j])
    return JJ


def Ising_generate(J_max,alpha,L,h_0):
    """generate Ising hamiltonian"""
    Jzz = J_zz_gen(J_max,alpha,L)  # long-range terms in OBC
    #Jzz = [[J_max,i,(i+1)%L] for i in range(L)]
    hx = [[0.5*h_0,i] for i in range(0,L)] 
    #hz = [[0.5*J_max,i] for i in range(0,L)]
    static = [["zz",Jzz],["x",hx]]
    dynamic = []
    basis = spin_basis_1d(L)
    #basis = spin_basis_1d(L,zblock=(-1)**(L//2))
    no_checks = {"check_herm":False,"check_pcon":False,"check_symm":False}
    H_ising=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks) #set ising hamiltonian 
    return H_ising  


def Hint(h,L):
    """
    interaction hamiltonian in formule of fidelity susceptibility
    """
    hx = [[0.5*h,i] for i in range(0,L)] 
    static = [["x",hx]]
    dynamic = []
    basis = spin_basis_1d(L)
    #basis = spin_basis_1d(L,zblock=(-1)**(L//2))
    no_checks = {"check_herm":False,"check_pcon":False,"check_symm":False}
    H_i=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks) #set ising hamiltonian 
    return H_i


n_real = 400 # number of realizations
L = 10 # number of sites
J_max = 1 # nearest-neighbor interaction strength
alpha = 2 # long-range exponent
step = 0.4 # step
basis = spin_basis_1d(L) # ising basis without symmetry
#basis = spin_basis_1d(L,zblock=(-1)**(L//2)) # ising basis with spin-inversion symmetry
#neel = findneel(L) # neel state |1010101010>

def get_FS(psi,E,Hmt):
    """
    for given spectrum and set of eigenstates, compute GS Fidelity susceptibility
    """
    chi = 0
    for n in range(1,len(E)):
        trm = abs(np.vdot(psi[n],Hmt.dot(psi[0]))) ** 2 / (E[n] - E[0]) ** 2
        chi += trm
    return chi


Dlist = np.linspace(1,15,125)
Blist = [1,2,4,8]
Fisher = []
errorbar = []    
for B in Blist:    
    F = [] 
    error_F = []
    for D in Dlist :
        H_int = Hint(1,L)
        qfi = []
        for i in range(n_real):
            Hamitonian = Ising_generate(J_max,alpha,L,B) + D * real_generate()
            E,V = Hamitonian.eigh()
            V = V.T
            qfi.append(4*get_FS(V,E,H_int))
        fs = np.average(np.array(qfi))
        F.append(fs)
        error_F.append(np.std(np.array(qfi)))
    Fisher.append(F)
    errorbar.append(error_F)


fo = open("fisher_information.txt","w")
for i in range(len(Blist)):    
    fo.write("## N=10,D in (1,15), B/J=")
    fo.write(str(Blist[i]))
    fo.write("\n[")
    for f in Fisher[i]:
        fo.write(str(f))
        fo.write(",")
    fo.write("]\n")
fo.close()

fo = open("errorbar_fisher_information.txt","w")
for i in range(len(Blist)):    
    fo.write("## N=10,D in (1,15), B/J=")
    fo.write(str(Blist[i]))
    fo.write("\n[")
    for e in errorbar[i]:
        fo.write(str(e))
        fo.write(",")
    fo.write("]\n")
fo.close()
