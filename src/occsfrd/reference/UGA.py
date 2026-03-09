import numpy as np
import string
from math import sqrt, factorial
from pyscf import fci#, gto, scf, ao2mo
from occsfrd.wick import operator, index

# class CSF:
#     def __init__(self, mol_, mf_, spin, projSpin, d):
#         self.mol = mol
#         self.mo_coeff = mf_.mo_coeff
# #        self.mo_occ = mf_.mo_occ
#         self.S = spin
#         self.M = projSpin
#         self.dvec = d

#    def couple_new_orbital(case):
#    if case == 0:
#    elif case == 1:
#        self.S = self.S + 0.5
#    elif case == 2:
#        self.S = self.S - 0.5
#    elif case == 3:

#    def get_CIExpansion(self):



# E001 = cisolver.energy(h1[0,0], eri[0,0], np.array([1.]),1,0) + Enuc
# E010 = cisolver.energy(h1[0,0], eri[0,0], np.array([1.]),1,1) + Enuc
# E100 = cisolver.energy(h1[0,0], eri[0,0], np.array([1.]),1,2) + Enuc
# E101 = cisolver.energy(h1, eri, np.array([1.,0.,0.,0.]),2,2) + Enuc
# TripE020 = cisolver.energy(h1, eri, np.array([0.,1/sqrt(2),-1/sqrt(2),0.]),2,2) + Enuc
# SingE020 = cisolver.energy(h1, eri, np.array([0.,1/sqrt(2),1/sqrt(2),0.]),2,2) + Enuc

# def PartialCoreH(h1, norb):
#     """
#     Restrict a given 1-electron Hamiltonian to only the first norb orbitals
#     """
#     return h1[:norb,:norb]

# def PartialERI(eri, norb):
#     """
#     Restrict a given 2-electron Hamiltonian to only the first norb orbitals
#     """
#     n = int(norb * (norb + 1) / 2)
#     return eri[:n,:n]

# def PartialEnergy(cisolver_, h1ematrix, erimatrix, CIExpansion, norb, nelec, Enuc_):
#     """
#     Energy of a given CIExpansion, restricted to only the first norb orbitals
#     """
#     return cisolver_.energy(PartialCoreH(h1ematrix, norb), PartialERI(erimatrix, norb), CIExpansion, norb, nelec) + Enuc_

def CGCoeff(Sold, Mold, s, m, S, M):
    '''
    General Clebsh--Gordan coefficient for addition of |s,m> orbital to |Sold, Mold> state to generate |S,M> state.
    '''
    if (Mold + m == M):
        A = sqrt((2 * S + 1) * factorial(int(S + Sold - s)) * factorial(int(S - Sold + s)) * factorial(int(Sold + s - S)) / factorial(int(Sold + s + S + 1)))
        B = sqrt(factorial(int(S + M)) * factorial(int(S - M)) * factorial(int(Sold + Mold)) * factorial(int(Sold - Mold)) * factorial(int(s + m)) * factorial(int(s - m)))
        C = 0
        for k in range(int(max(Sold + s - S, Sold - Mold, s + m)) + 1):
            if (Sold + s - S < k) or (Sold - Mold < k) or (s + m < k) or (S - s + Mold < -k) or (S - Sold - m < -k):
                C = C
            else:
                C += pow(-1, k) / (factorial(k) * factorial(int(Sold + s - S - k)) * factorial(int(Sold - Mold - k)) * factorial(int(s + m - k)) * factorial(int(S - s + Mold + k)) * factorial(int(S - Sold - m + k)))
        return A * B * C
    return 0

#cisolver.kernel()

# class Determinant:
#     def __init__(self, mol, mf):
#         self.mol = mol
#         self.mo_coeff = mf.mo_coeff
#         self.mo_occ = mf.mo_occ
    
#     def get_1e_ints(self):
#         gto.

#     def get_energy(self):


# #Energy of a linear combination of orthonormal Slater determinants
# def LCSD_energy(coeffs, dets):
#     E = 0
#     for d in range(len(coeffs)):
#         c = coeffs[d]
#         E += np.conjugate(c) * c * dets[d].get_energy()
#     return E

# def determinant_to_cibasis(stringa, stringb, neleca, nelecb, norb):
#     addra = fci.cistring.str2addr(norb, neleca, stringa)
#     addrb = fci.cistring.str2addr(norb, nelecb, stringb)

#     nstrsa = fci.cistring.num_strings(norb, neleca)
#     nstrsb = fci.cistring.num_strings(norb, nelecb)

#     cibasis = np.zeros((nstrsa, nstrsb))
#     cibasis[addra, addrb] = 1

#     return cibasis

def gen_CSF_summand(S, M, orbIndices, dvec, mvec):
    """
    Generate one term (corresponding to a given mvec)
    in the summation of a CSF of given S, M, and dvec
    by the Yamanouchi--Kotani scheme
    """
# Set up vacuum state
    # neleca, nelecb = 0, 0
    # state = np.array([[1.]])
    operatorList = []

    i = len(dvec)
    Si = S
    Mi = M
    CGProd = 1

# Populate orbitals using second-quantized algebra
    while i > 0:
        i = i - 1
        mi = mvec[i]
        case = dvec[i]
        if case == 0:
            ref = ref
            CGProd = CGProd * CGCoeff(Si, Mi, 0, 0, Si, Mi)
        elif case == 1:
            if mi == 0.5:
                # state = fci.addons.cre_a(state, norbs, (neleca, nelecb), i)
                operatorList = [operator.BasicOperator(orbIndices[i], creation_annihilation_=True, spin_=True)] + operatorList
                CGProd = CGProd * CGCoeff(Si - 0.5, Mi - 0.5, 0.5, 0.5, Si, Mi)
                # neleca += 1
                Mi = Mi - 0.5
            elif mi == -0.5:
                # state = fci.addons.cre_b(state, norbs, (neleca, nelecb), i)
                operatorList = [operator.BasicOperator(orbIndices[i], creation_annihilation_=True, spin_=False)] + operatorList
                CGProd = CGProd * CGCoeff(Si - 0.5, Mi + 0.5, 0.5, -0.5, Si, Mi)
                # nelecb += 1
                Mi = Mi + 0.5
            Si = Si - 0.5
        elif case == 2:
            if mi == 0.5:
                # state = fci.addons.cre_a(state, norbs, (neleca, nelecb), i)
                operatorList = [operator.BasicOperator(orbIndices[i], creation_annihilation_=True, spin_=True)] + operatorList
                CGProd = CGProd * CGCoeff(Si + 0.5, Mi - 0.5, 0.5, 0.5, Si, Mi)
                # neleca += 1
                Mi = Mi - 0.5
            elif mi == -0.5:
                # state = fci.addons.cre_b(state, norbs, (neleca, nelecb), i)
                operatorList = [operator.BasicOperator(orbIndices[i], creation_annihilation_=True, spin_=False)] + operatorList
                CGProd = CGProd * CGCoeff(Si + 0.5, Mi + 0.5, 0.5, -0.5, Si, Mi)
                # nelecb += 1
                Mi = Mi + 0.5
            Si = Si + 0.5
        elif case == 3:
            # state = fci.addons.cre_b(state, norbs, (neleca, nelecb), i)
            # nelecb += 1
            # state = fci.addons.cre_a(state, norbs, (neleca, nelecb), i)
            # neleca += 1
            # state = state
            operatorList = [operator.BasicOperator(orbIndices[i], creation_annihilation_=True, spin_=True), operator.BasicOperator(orbIndices[i], creation_annihilation_=True, spin_=False)] + operatorList
            CGProd = CGProd * CGCoeff(Si, Mi, 0, 0, Si, Mi)

    return operator.OperatorProduct(operatorList, CGProd)
    return state * CGProd

def gen_mvec_list(M, dvec):
    """
    Generate list of mvecs commensurate with the given dvec and the total M
    """
    nocc = len(dvec)
    mvecs = np.reshape(np.zeros(len(dvec)), (1, -1))

    for i in range(nocc):
        if dvec[i] == 1 or dvec[i] == 2:
            newmvecs = mvecs.copy()
            for v in range(len(mvecs)):
                mvecs[v,i] = 0.5
                newmvecs[v,i] = -0.5
            mvecs = np.concatenate((mvecs, newmvecs))
        elif dvec[i] == 0 or dvec[i] == 3:
            mvecs = mvecs
    vecs_to_delete = []
    for v in range(len(mvecs)):
        sum_mi = 0
        si = 0
        for i in range(len(mvecs[v])):
            sum_mi = sum_mi + mvecs[v,i]
            if dvec[i] == 1:
                si = si + 0.5
            elif dvec[i] == 2:
                si = si - 0.5
            if abs(sum_mi) > si:
                vecs_to_delete.append(v)
        if np.sum(mvecs[v]) != M and v not in vecs_to_delete:
            vecs_to_delete.append(v)
    return np.delete(mvecs, vecs_to_delete, axis = 0)

def gen_CSF(S, M, orbIndices, dvec):
    '''
    '''
    mvecList = gen_mvec_list(M, dvec)

    CSF = sum([gen_CSF_summand(S, M, orbIndices, dvec, mvecList[i]) for i in range(len(mvecList))])

    # CSF = operator.OperatorSum([])
    # for i in range(len(mvecList)):
    #     mvec = mvecList[i]
    #     CSF = CSF + gen_CSF_summand(S, M, orbIndices, dvec, mvec)

    return CSF

def multipleCSF_ref(nActiveOrbs, GelfandTsetlinCSFlist, activeOrbitalIndices_=None):
    '''
    Generate the operators to construct an open shell reference state out of a Fermi vacuum.
    Each (S, M, dvec, prefactor) tuple in GelfandTsetlinCSFlist specifies a CSF to be summed in the reference.
    '''
    activeOrbitalIndices = []
    if activeOrbitalIndices_ is None:
        activeOrbitalIndices = [index.SpecificOrbitalIndex(string.ascii_lowercase[i+8], occupiedInVacuum=False, active=True, specificIndexValue=i) for i in range(nActiveOrbs)]
    else:
        assert len(activeOrbitalIndices_) == nActiveOrbs
        activeOrbitalIndices = activeOrbitalIndices_
    # ref = operator.OperatorSum([])
    # for i in range(len(GelfandTsetlinCSFlist)):
    #     S, M, dvec, prefactor = GelfandTsetlinCSFlist[i]
    #     ref = ref + prefactor * gen_CSF(S, M, activeOrbitalIndices, dvec)

    ref = sum([GelfandTsetlinCSFlist[i][3] * gen_CSF(GelfandTsetlinCSFlist[i][0], GelfandTsetlinCSFlist[i][1], activeOrbitalIndices, GelfandTsetlinCSFlist[i][2]) for i in range(len(GelfandTsetlinCSFlist))])

    return {"reference": ref, "indices": activeOrbitalIndices}

def singleCSF_ref(nActiveOrbs, GelfandTsetlinCSF, prefactor=1., activeOrbitalIndices_=None):
    activeOrbitalIndices = []
    if activeOrbitalIndices_ is None:
        activeOrbitalIndices = [index.SpecificOrbitalIndex(string.ascii_lowercase[i+8], occupiedInVacuum=False, active=True, specificIndexValue=i) for i in range(nActiveOrbs)]
    else:
        assert len(activeOrbitalIndices_) == nActiveOrbs
        activeOrbitalIndices = activeOrbitalIndices_
    S, M, dvec = GelfandTsetlinCSF
    ref = prefactor * gen_CSF(S, M, activeOrbitalIndices, dvec)

    return {"reference": ref, "indices": activeOrbitalIndices}