
from quspin.operators import hamiltonian,quantum_operator
from quspin.basis import spin_basis_1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

#### Chaobao Gai zhe ge fieldstren
fieldstren = 9

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
    #Jzz = [[0.25*J_max,i,(i+1)%L] for i in range(L)]
    hx = [[0.5*h_0,i] for i in range(0,L)] 
    #hz = [[0.5*J_max,i] for i in range(0,L)]
    static = [["zz",Jzz],["x",hx]]#,["z",hz]]
    dynamic = []
    basis = spin_basis_1d(L,zblock=(-1)**(L//2))
    no_checks = {"check_herm":False,"check_pcon":False,"check_symm":False}
    H_ising=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks) #set ising hamiltonian 
    return H_ising  

def correlator(L):
    """
    generate accessible measurement observable 
    Jx0 = 0.5 * \sum_n \sigma_n^x 
    """
    basis = spin_basis_1d(L)
    Jxx = []
    for i in range(L):
        for j in range(L):
            Jxx.append([(-1)**(i+j+2),i,j])
    xlist = [["xx",Jxx]]
    operator_dict = dict(x0=xlist)
    Jx0 = quantum_operator(operator_dict,basis=basis,check_symm=False,check_herm=False)
    Jx0 = Jx0.tohamiltonian()
    return Jx0

def M_generate(L,index,p):
    """
    generate accessible measurement observable 
    index = "x","y","z"
    p = +-1
    M(L,"x",1) = 0.5 * \sum_n \sigma_n^x 
    M(L,"x",-1) = 0.5 * \sum_n (-1)**n * \sigma_n^x 
    """
    basis = spin_basis_1d(L,zblock=(-1)**(L//2))
    if p==1:
        J = [[0.5,i] for i in range(L)]
        if index=="x":
            list = [["x",J]]
        elif index=="y":
            list = [["y",J]]
        elif index=="z":
            list = [["z",J]]
        operator_dict = dict(list0=list)
    elif p==-1:
        J = [[0.5*(-1)**(i+1),i] for i in range(L)]
        if index=="x":
            list = [["x",J]]
        elif index=="y":
            list = [["y",J]]
        elif index=="z":
            list = [["z",J]]
        operator_dict = dict(list0=list)
    J = quantum_operator(operator_dict,basis=basis,check_symm=False,check_herm=False)
    J = J.tohamiltonian({'list0':1})
    return J



def derivative_w(v1,v2,M,dw=0.01):
    """
    for quantum operator M, compute its derivative to parameter M
    """
    dM_dw = [] 
    for i in M:
        M2 = i.expt_value(v2)
        M1 = i.expt_value(v1)
        dM_dw.append((M2-M1)/dw)
    return dM_dw

def QFI(v0,v1,v2,times):
    """
    compute quantum fisher information for each time in np.ndarray times
    """
    qfi = 4 * (QFI_term1(v1,v2,times,dw=0.02) - QFI_term2(v0,v1,v2,times,dw=0.02))
    return np.real(qfi)

def QFI_term1(v1,v2,times,dw=0.005):
    qfi1 = []
    dv_dw = (v2 - v1)/dw
    for i in range(len(times)):
        qfi1.append(np.vdot(dv_dw[:,i],dv_dw[:,i]))
    return np.array(qfi1)

def QFI_term2(v0,v1,v2,times,dw=0.005):
    qfi2 = []
    dv_dw = (v2 - v1)/dw
    for i in range(len(times)):
        term = np.vdot(v0[:,i],dv_dw[:,i])
        qfi2.append(abs(term) ** 2)
    return np.array(qfi2)


def QFI_anotherway(v,times):
    qfi1 = []
    for i in range(len(times)):    
        term = M(L).quant_fluct(v[:,i])
        qfi1.append(np.real(term))

    return np.array(qfi1)


def expt_M(s,v):
    """
    The vector of expectation values <M> where:
            M = [Jx0, Jy0, Jz0, Jx1, Jy1, Jz1]
        where 
            Jx0 = 0.5 * \sum_n \sigma_n^x 
        and 
            Jx1 = 0.5 * \sum_n (-1)**n * \sigma_n^x
        etc.
    """
    out = []
    for i in s:
        out.append(i.expt_value(v))
    return np.array(out)

def covariance(c,s,v):
    """
    The covariance matrix whose matrix elements are
            C_ij = Cov(M_i, M_j)
                 = 0.5*<(M_i*M_j + M_j*M_i)> - <M_i><M_j>
        where M is the vector of operators
            M = [Jx0, Jy0, Jz0, Jx1, Jy1, Jz1]
    """
    out = []
    for i in range(len(c)):
        lineele = []
        for j in range(len(c[i])):
            ele = c[i,j].expt_value(v) - s[i].expt_value(v) * s[j].expt_value(v)
            lineele.append(ele)
        out.append(lineele)
    return np.array(out)

def estimation_error(Theta_list, M, dM_dw, C, epsilon=1e-9):

    """
    Inputs
    ------

    Theta_list : np.array
        The list of 5 angles that define a unit vector in 6-dimensional
        space.

    M : np.array
        The vector of expectation values <M> where:
            M = [Jx0, Jy0, Jz0, Jx1, Jy1, Jz1]
        where 
            Jx0 = 0.5 * \sum_n \sigma_n^x 
        and 
            Jx1 = 0.5 * \sum_n (-1)**n * \sigma_n^x
        etc.

    dM_dw : np.array
        The vector of derivatives with respect to the unknown parameter,
            <dM/dw> = (d/dw) <[Jx0, Jy0, Jz0, Jx1, Jy1, Jz1]>

    C : np.array
        The covariance matrix whose matrix elements are
            C_ij = Cov(M_i, M_j)
                 = 0.5*<(M_i*M_j + M_j*M_i)> - <M_i><M_j>
        where M is the vector of operators
            M = [Jx0, Jy0, Jz0, Jx1, Jy1, Jz1]

    Output
    ------

    The estimation error, given by the expression

        est_error = np.dot(r, C @ r) / (np.dot(r, dM_dw)**2 + epsilon)

    as well as the gradients of the estimation error, with respect to
    the angles in Theta_list.
    """

    # unit vector
    r = np.array([np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1])*
                    np.sin(Theta_list[0]),
                  np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1])*
                    np.cos(Theta_list[0]),
                  np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.cos(Theta_list[1]),
                  np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.cos(Theta_list[2]),
                  np.sin(Theta_list[4])*np.cos(Theta_list[3]),
                  np.cos(Theta_list[4])])

    dr_dTheta0 = np.array([np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1])*
                    np.cos(Theta_list[0]),
                  -np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1])*
                    np.sin(Theta_list[0]),
                  0,
                  0,
                  0,
                  0])

    dr_dTheta1 = np.array([np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.cos(Theta_list[1])*
                    np.sin(Theta_list[0]),
                  np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.cos(Theta_list[1])*
                    np.cos(Theta_list[0]),
                  -np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1]),
                  0,
                  0,
                  0])

    dr_dTheta2 = np.array([np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.cos(Theta_list[2])*np.sin(Theta_list[1])*
                    np.sin(Theta_list[0]),
                  np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.cos(Theta_list[2])*np.sin(Theta_list[1])*
                    np.cos(Theta_list[0]),
                  np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.cos(Theta_list[2])*np.cos(Theta_list[1]),
                  -np.sin(Theta_list[4])*np.sin(Theta_list[3])*
                    np.sin(Theta_list[2]),
                  0,
                  0])

    dr_dTheta3 = np.array([np.sin(Theta_list[4])*np.cos(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1])*
                    np.sin(Theta_list[0]),
                  np.sin(Theta_list[4])*np.cos(Theta_list[3])*
                    np.sin(Theta_list[2])*np.sin(Theta_list[1])*
                    np.cos(Theta_list[0]),
                  np.sin(Theta_list[4])*np.cos(Theta_list[3])*
                    np.sin(Theta_list[2])*np.cos(Theta_list[1]),
                  np.sin(Theta_list[4])*np.cos(Theta_list[3])*
                    np.cos(Theta_list[2]),
                  -np.sin(Theta_list[4])*np.sin(Theta_list[3]),
                  0])

    dr_dTheta4 = np.array([
        np.cos(Theta_list[4])*np.sin(Theta_list[3])*
          np.sin(Theta_list[2])*np.sin(Theta_list[1])*
          np.sin(Theta_list[0]),
        np.cos(Theta_list[4])*np.sin(Theta_list[3])*
          np.sin(Theta_list[2])*np.sin(Theta_list[1])*
          np.cos(Theta_list[0]),
        np.cos(Theta_list[4])*np.sin(Theta_list[3])*
          np.sin(Theta_list[2])*np.cos(Theta_list[1]),
        np.cos(Theta_list[4])*np.sin(Theta_list[3])*
          np.cos(Theta_list[2]),
        np.cos(Theta_list[4])*np.cos(Theta_list[3]),
        -np.sin(Theta_list[4])])

    est_error = abs(np.dot(r, C @ r)) / abs(np.dot(r, dM_dw)**2 + epsilon)

    derror_dTheta0 = (np.dot(dr_dTheta0, C@r) +
                      np.dot(r, C@dr_dTheta0)) /\
                     (np.dot(r, dM_dw)**2 + epsilon) - \
                     2*np.dot(r, C@r)*np.dot(dr_dTheta0, dM_dw) / \
                     (np.dot(r, dM_dw)**3  + epsilon)
    derror_dTheta1 = (np.dot(dr_dTheta1, C@r) +
                      np.dot(r, C@dr_dTheta1)) /\
                     (np.dot(r, dM_dw)**2  + epsilon) - \
                     2*np.dot(r, C@r)*np.dot(dr_dTheta1, dM_dw) / \
                     (np.dot(r, dM_dw)**3  + epsilon)
    derror_dTheta2 = (np.dot(dr_dTheta2, C@r) +
                      np.dot(r, C@dr_dTheta2)) /\
                     (np.dot(r, dM_dw)**2  + epsilon) - \
                     2*np.dot(r, C@r)*np.dot(dr_dTheta2, dM_dw) / \
                     (np.dot(r, dM_dw)**3  + epsilon)
    derror_dTheta3 = (np.dot(dr_dTheta3, C@r) +
                      np.dot(r, C@dr_dTheta3)) /\
                     (np.dot(r, dM_dw)**2 + epsilon) - \
                     2*np.dot(r, C@r)*np.dot(dr_dTheta3, dM_dw) / \
                     (np.dot(r, dM_dw)**3  + epsilon)
    derror_dTheta4 = (np.dot(dr_dTheta4, C@r) +
                      np.dot(r, C@dr_dTheta4)) /\
                     (np.dot(r, dM_dw)**2  + epsilon) - \
                     2*np.dot(r, C@r)*np.dot(dr_dTheta4, dM_dw) / \
                     (np.dot(r, dM_dw)**3  + epsilon)

    return np.real(est_error)#, \
        #np.real(np.array([derror_dTheta0, derror_dTheta1, derror_dTheta2,
        #          derror_dTheta3, derror_dTheta4]))


def optimised_error(M, dM_dw, C):
    
    theta_init = [2*np.pi*np.random.random() for i in range(5)]

    args = (M, dM_dw, C, 1e-9)
    res = so.minimize(estimation_error, theta_init, args=args)#,
                           #jac=True)

    return res.fun

def set_time(f,d):
    t1 = np.arange(0.05,25,0.05)
    t2 = np.logspace(-1,2.3,300)
    t3 = np.arange(60,170,1)
    t4 = np.arange(65,220,1)
    if f==1:
        if d<5.6:
            return t1
        else:
            return t4
    if f==1.5:
        if d<5.2:
            return t1
        else:
            return t4
    if f==2:
        if d<4.7:
            return t1
        elif d<5.1:
            return t2
        else:
            return t3
    if f==2.5:
        if d<3.6:
            return t1
        elif d<4.2:
            return t2
        else:
            return t3
    if f==3:
        if d<3.3:
            return t1
        elif d<3.6:
            return t2
        else:
            return t3
    if f==3.5:
        if d<3.2:
            return t1
        elif d<3.7:
            return t2
        else:
            return t3
    if f==4:
        if d<2.7:
            return t1
        elif d<3.7:
            return t2
        else:
            return t4
    if f==4.5:
        if d<3.2:
            return t1
        elif d<3.7:
            return t2
        else:
            return t4
    if f==5:
        if d<3.2:
            return t1
        elif d<3.7:
            return t2
        else:
            return t4
    


def findneel(L):
    """
    Hamiltonian, the corresponding eigenstate of its maximum
    eigenvalue is the neel state
    """
    hx = [[(-1)**(i+1),i] for i in range(0,L)] 
    static = [["x",hx]]
    dynamic = []
    #basis = spin_basis_1d(L)
    basis = spin_basis_1d(L,zblock=(-1)**(L//2))
    no_checks = {"check_herm":False,"check_pcon":False,"check_symm":False}
    H_neel=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks) #set ising hamiltonian 
    Emin,Emax = H_neel.eigsh(k=2,which="BE",return_eigenvectors=False)
    Em,neelstate = H_neel.eigsh(k=1,sigma=Emax+0.001,maxiter=1E4)
    neelstate = neelstate.reshape((-1,))
    return neelstate

#### set parametres and basis
n_real = 200 # number of realizations
L = 10 # number of sites
J_max = 1 # nearest-neighbor interaction strength
h_0 = 4.0  # transversed static magnetic field
h_dis_max = 8.  # maximum disorder strength
alpha = 2 # long-range exponent
step = 0.4 # step
basis = spin_basis_1d(L,zblock=(-1)**(L//2)) # ising basis without symmetry
neel = findneel(L) # neel state |1010101010>




def state_output(J_max,h_dis,alpha,L,times,w,psi0 = neel,dw=0.01):
    """
    input:
        psi0: initial state
        J_max: maximum long-range coupling strength
        h_dis: disorder strength
        w: transverse field strength
        times: np.array of the times we want to calculate the evolvong state
    output: 
        |psi_w> : np.array 
        |psi_w+dw>: np.array
    """
    H_dis = h_dis * real_generate()
    H1 = Ising_generate(J_max,alpha,L,w) + H_dis
    H2= Ising_generate(J_max,alpha,L,w+dw) + H_dis
    psi_w1 = H1.evolve(psi0,0,times)
    psi_w2 = H2.evolve(psi0,0,times)
    return psi_w1,psi_w2
    

#### generate operator lists that needs to be computed
# spin sums
M = []
M.append(M_generate(L,"x",1))
M.append(M_generate(L,"y",1))
M.append(M_generate(L,"z",1))
M.append(M_generate(L,"x",-1))
M.append(M_generate(L,"y",-1))
M.append(M_generate(L,"z",-1))
# covariance matrix
C_left = []
for i in range(6):
     a = []
     for j in range(6):
          a.append(0.5*(M[i].dot(M[j])+M[j].dot(M[i])))
     C_left.append(a)
C_left = np.array(C_left)

#### big is coming!!!
opterror = []
opttime = []
errorbar = []
hlist = np.linspace(0.5,4.7,9)
for h in hlist:
    times_test = np.linspace(0.1,2.5,80)
    ####
    ee = []
    real = 200
    error_iter = []
    #### average over n_real realisations
    for j in range(real):
        e = []
        v1,v2 = state_output(1,h,alpha,L,times_test,fieldstren,psi0=neel,dw=0.01)
        error_t = []
        #### calculate MOM error for each time
        for t in range(len(times_test)):
            dM_dw = derivative_w(v1[:,t],v2[:,t],M,dw=0.01) / np.sqrt(times_test[t])
            avr_M = expt_M(M,v1[:,t])
            C = covariance(C_left,M,v1[:,t])
            error = optimised_error(avr_M, dM_dw, C)
            error_t.append(np.sqrt(error))
        error_iter.append(error_t)
    error_iter = np.array(error_iter)
    for i in range(len(times_test)):
        ee.append(np.average(error_iter[:,i]))
    #### record optimal error&time
    opterror.append(min(ee))
    opttime.append(times_test[ee.index(min(ee))])
    errorbar.append(np.std(error_iter[:,ee.index(min(ee))]))
opttime = 1/np.array(opttime)

### plot figure
plt.plot(hlist,opttime,label=r'N=10')
plt.xlabel(r'$D/J_{max}$')
plt.ylabel(r'$1/t_*$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(str(fieldstren)+'.time_vs_disorder.jpg')
plt.show()


fo = open(str(fieldstren)+"opterror1.txt","w")
fo.write("## N=10,D in (0.5,10), B/J=")
fo.write(str(fieldstren))
fo.write("## \n[")
for i in opterror:
    fo.write(str(i))
    fo.write(",")
fo.write("]")
fo.close()

fo = open(str(fieldstren)+"opttime1.txt","w")
fo.write("## N=10,D in (0.5,10), B/J=")
fo.write(str(fieldstren))
fo.write("## \n[")
fo.write("[")
for i in opttime:
    fo.write(str(i))
    fo.write(",")
fo.write("]")
fo.close()

fo = open(str(fieldstren)+"errorbar1.txt","w")
fo.write("## N=10,D in (0.5,10), B/J=")
fo.write(str(fieldstren))
fo.write("## \n[")
fo.write("[")
for i in errorbar:
    fo.write(str(i))
    fo.write(",")
fo.write("]")
fo.close()