# -*- coding: utf-8 -*-
"""
Created on Fri May 30 14:40:21 2025

@author: Xin Lu
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpi4py import MPI
import time
import sys
import os
import warnings
from numpy import (
    load, savez, savez_compressed,
    array, linspace, arange, empty, zeros, ones, eye, full, identity, matrix, matmul,
    hstack, vstack, stack, concatenate, sort, argsort, where, all, any, expand_dims,
    tensordot, einsum, dot, vdot, inner, kron, cross, amin, amax, ascontiguousarray,
    trace, transpose, conj, real, imag, diag, sum, prod, diagonal, fill_diagonal, roll,
    around, absolute, angle, pi, sqrt, exp, log, sin, cos, tan, heaviside, nonzero, binary_repr,
)
from numpy.linalg import eigh, eigvalsh, det, inv, norm, eig, eigvals
from scipy.sparse.linalg import eigs,eigsh
from scipy.sparse import csr_matrix
from scipy.special import binom
from matplotlib.pyplot import subplots, figure, plot, imshow, scatter
from matplotlib import cm
from numba import njit,types
from numba.typed import Dict



Nphi = int(sys.argv[1])
p = int(sys.argv[2])
q = int(sys.argv[3])
N = Nphi//q
Ne = N*p

if p*q*(Ne-1) % 2 == 0:
    s0=0; t0=0
elif p*q*(Ne-1) % 2 == 1:
    s0=N//2; t0=N//2

ratio_rect = float(sys.argv[4]) # a/b
numiteration = int(sys.argv[5])
num = 5
algo = int(sys.argv[6])

if algo == 0:
    algostr = '          Trivial Lanczos'
elif algo == 1:
    algostr = 'Partial Reorthogonalization'
elif algo == 2:
    algostr = 'Part. Reortho. with Cauchy Convergence'
elif algo == 3:
    algostr = 'Part. Reortho. with Residual Value Monitoring'
elif algo == -1:
    algostr = 'Arnoldi'
elif algo == -2:
    algostr = 'Full matrix'

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()



@njit
def occupation(conf,i):
    return (conf&(1<<i))>>i

def binconf(c,L):
    "binary representation of a conf integer with zeroth site on the right"
    return binary_repr(c,L)

def bitcount(c,L):
    "counts number of 1 in binary representation of a conf integer"
    return binary_repr(c,L).count("1")

@njit
def count_ones_between_bits(c, start, end): # 包含start和end位，取值从0到L-1
    "counts number of 1 in binary representation of a conf integer between start and end position (endpoints included)"
    if start>end:
        return 0
    else:
        # Step 1: 创建掩码，提取位置i到j之间的位
        mask = (1 << (end - start + 1)) - 1 # 创建一个有 end - start + 1 个 1 的掩码
        c &= (mask << start) # 将 n 右移 i 位并与掩码按位与，提取 start 到 end 之间的位,包含start和end位: start=2,end=6, mask << start = 0..001111100
        # Step 2: Use parallel bit counting techniques to count ones
        S = [1,2,4,8,16,32]
        B = [0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF, 0x00000000FFFFFFFF]
        count = c - ((c >> S[0]) & B[0])
        count = ((count >> S[1]) & B[1]) + (count & B[1])
        count = ((count >> S[2]) + count) & B[2]
        count = ((count >> S[3]) + count) & B[3]
        count = ((count >> S[4]) + count) & B[4]
        count = ((count >> S[5]) + count) & B[5]
        return count

@njit
def count_ones_from(c, start):
    "counts number of 1 in binary representation of a conf integer from start to the last position (endpoints included)"
    c = c >> start
    S = [1,2,4,8,16,32]
    B = [0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF, 0x00000000FFFFFFFF]
    count = c - ((c >> S[0]) & B[0])
    count = ((count >> S[1]) & B[1]) + (count & B[1])
    count = ((count >> S[2]) + count) & B[2]
    count = ((count >> S[3]) + count) & B[3]
    count = ((count >> S[4]) + count) & B[4]
    count = ((count >> S[5]) + count) & B[5]
    return count

@njit
def translate_to_right(conf,q): # (Tq)^(-1) in Eq.(83)
    "translation of q sites to the right"
    firstq = conf&(2**(q)-1)
    return (conf>>q)|(firstq<<(Nphi-q))

@njit
def translate_to_right_with_sign(conf,q): # Eq.(92)
    "translation of q sites to the right and return also the sign change"
    firstq = conf&(2**(q)-1)
    NeL = count_ones_from(conf,q)
    NeR = count_ones_from(firstq,0)
    if (NeL*NeR)%2==0:
        sign = 1
    else:
        sign = -1
    newconf = (conf>>q)|(firstq<<(Nphi-q))
    return newconf, sign

@njit
def representative(conf): # Eq.(89) and Eq.(92)
    "generate the representative for a given conf and its associated times of translations"
    rep = conf; at = conf; la = 0; sign_at = 1
    sign_a = 1
    for k in range(N):
        at, sign = translate_to_right_with_sign(at,q)
        sign_at *= sign   
        if (at < rep):
            rep = at; la = k+1; sign_a = sign_at 
    return rep, la, sign_a
            

@njit
def checkstate_with_sign(conf,s): # Eq.(94)
    "check if conf (|a>) is the representative for a given s (s could be invalid) then return Ra"
    at = conf # |a>
    signt = 1
    for k in range(N):
        at, sign = translate_to_right_with_sign(at,q)
        signt *= sign
        if at < conf:
            return -1,0
        elif (at == conf):
            Qa = N/(k+1)
            if signt > 0:
                if (np.mod(s,Qa) != 0): 
                    return -1,0
                else:
                    return k+1,signt
            elif signt < 0:
                if (np.mod(2*s-Qa,2*Qa) != 0):
                    return -1,0
                else:
                    return k+1,signt



hilbertsize = int(binom(Nphi,Ne))
basis_rank = []

@njit
def totmomentum(c):
    "calculate total momentum of a conf modulo Nphi"
    jtot = 0
    for j in range(Nphi):
        if c & (1 << j):
            jtot += j
    return jtot%Nphi

def fillhilbert_with_sign(conf,s,t): 
    global basis_rank
    tc = totmomentum(conf)
    Ra, sign_a = checkstate_with_sign(conf,s)
    if tc == t and Ra > 0:
        basis_rank.append((conf,Ra))


def create_Hilbert(s,t): # MPI
    timeh0 = time.time()
    global basis_rank
    basis_rank = [] # generated basis in each process
    def SortByQn(Ls,kmin,kmax):
        res = {}
        for conf in range(2**kmin-1,((2**kmax-1)<<(Ls-kmax))+1):
            k = bitcount(conf,Ls)
            if kmin <= k <= kmax:
                if not k in res: 
                    res[k] = [conf]
                else: 
                    res[k].append(conf)
        return res

    LL = Nphi-Nphi//2
    LR = Nphi//2
    leftConfs  = SortByQn(LL,max(0,Ne-LR),min(Ne,LL))
    rightConfs = SortByQn(LR,max(0,Ne-LL),min(Ne,LR))
    for kl in leftConfs.keys():
        kr = Ne - kl
        if kr in rightConfs.keys():
            NconfL = len(leftConfs[kl])
            NconfR = len(rightConfs[kr])
            Nconf = NconfL*NconfR
            ########################### MPI ###########################
            # If the size of array to be parallelized *can not be divided* by the number of cores,
            # the array will be diveded into subsets with 2 types of size:
            # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
            subset_size,num_more=divmod(Nconf,size)
            indc_subsets=[range(Nconf)[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else range(Nconf)[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] 
            indc_subset=comm.scatter(indc_subsets,root=0) # scatter the index of conf with Ne particles
            ###########################################################
            for indc in indc_subset: # indcL*NconfR + indcR
                indcL = indc // NconfR
                indcR = indc % NconfR
                confL = leftConfs[kl][indcL]
                confR = rightConfs[kr][indcR]
                conf = (confL<<LR)^confR
                # fillhilbert(conf,s,t)
                fillhilbert_with_sign(conf,s,t)
    comm.barrier()
    timeh1 = time.time()
    if rank == 0:
        print(f'(s,t)=({s},{t}): confs are found for {(timeh1-timeh0):.2f}s.')
    ########################### MPI ###########################
    basis_rank_gather = comm.allgather(basis_rank)
    basis_merged = []
    for b in basis_rank_gather:
        basis_merged.extend(b)
    ###########################################################
    timeh2 = time.time()
    if rank == 0:
        print(f'Basis are merged for {(timeh2-timeh1):.2f}s.')
    if rank == 0:
        print(f"{len(basis_merged)} over {hilbertsize} configurations")
    del basis_rank
    return array(basis_merged,dtype=np.int64)


@njit
def generate_jlist(): # Eq.(82)
    jlist = []
    for j1 in range(Nphi):
        for j2 in range(j1+1,Nphi):
            for j3 in range(Nphi):
                for j4 in range(j3+1,Nphi):
                    momentum_consv = (j1+j2-j3-j4) % Nphi
                    if momentum_consv==0:
                        jlist.append([j1,j2,j3,j4])
    return array(jlist)
jlist = generate_jlist()

@njit
def generate_Vc(): # Eq.(74)
    j_Vc = []
    Vc = []
    tcut = Nphi*10
    scut = Nphi*10
    for j1 in range(Nphi):
        for j2 in range(Nphi):
            if j1!=j2:
                for j3 in range(Nphi):
                    for j4 in range(Nphi):
                        if j3!=j4:
                            momentum_consv = (j1+j2-j3-j4) % Nphi
                            if momentum_consv==0:
                                j_Vc.append((j1,j2,j3,j4))
                                Vq = 1j * 0.
                                for t in range(-tcut,tcut):
                                    if (j1-j4-t)%Nphi==0:
                                        for s in range(-scut,scut):
                                            if t!=0 or s!=0:
                                                qlBv = sqrt(2*pi/(Nphi*ratio_rect)) * array([s,ratio_rect*t])
                                                qlB = norm(qlBv)
                                                # Coulomb
                                                Vq += 1/2 * 1/(qlB * Nphi) * np.exp(-1/2*qlB**2) * np.exp(1j*2*pi*s/Nphi*(j1-j3)) # unit: e^2/(epsilon*lB)
                                                # Laughlin V1 potential
                                                # Vq += 1/2 * 1/(Nphi) *(1-qlB**2) * np.exp(-1/2*qlB**2) * np.exp(-1j*2*pi*s/Nphi*(j1-j3))# unit: e^2/(epsilon*lB)
                                Vc.append(Vq)
    return array(Vc),array(j_Vc)

@njit
def Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat): # Eq.(74)
    row, col, data = [], [], []
    subset_size = len(indb_subset)
    for nconf, indc in enumerate(indb_subset):
        if rank == 0 and (nconf==0 or (nconf+1)%1e5==0):
            print(nconf+1,'/',subset_size)
        conf,Ra = basis[indc] # |a>
        for i in range(len(jlist)):
            j1,j2,j3,j4 = jlist[i]
            sign = 1
            if occupation(conf,j3) & occupation(conf,j4):
                # j4
                sign *= (-1)**count_ones_from(conf,j4+1)
                newconf = conf & (~(1 << j4))
                # j3
                sign *= (-1)**count_ones_from(newconf,j3+1)
                newconf = newconf & (~(1 << j3))
                if (1-occupation(newconf,j1)) & (1-occupation(newconf,j2)):
                    # j2
                    sign *= (-1)**count_ones_from(newconf,j2+1)
                    newconf = newconf | (1 << j2)
                    # j1
                    sign *= (-1)**count_ones_from(newconf,j1+1)
                    newconf = newconf | (1 << j1)
                    Vq = Coulomb_mat[j1,j2,j3,j4] - Coulomb_mat[j2,j1,j3,j4] - Coulomb_mat[j1,j2,j4,j3] + Coulomb_mat[j2,j1,j4,j3] # Eq.(82)
                    rescoef = conj(Vq*sign) # ATT: what we need is <conf|V|newconf>, the above writing is for <newconf|V|conf>
                    if newconf != conf:
                        repb, la, sign_a = representative(newconf)
                        if repb in invbasis:
                            indc_b = invbasis[repb]
                            _, Rb = basis[indc_b]
                            rescoef *= np.exp(-1j*2*pi*s/N*la)*sqrt(Ra/Rb)*sign_a # Eq.(91) and Eq.(93)
                            if absolute(rescoef)>1e-14:
                                row.append(nconf)
                                col.append(indc_b)
                                data.append(rescoef)
                    else:
                        if absolute(rescoef)>1e-14:
                            row.append(nconf)
                            col.append(indc)
                            data.append(rescoef)
    return row,col,data

def fullmatrix(indb_subset,basis,invbasis,s,num):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    if rank==0:
        print('Nb. of Non-zero elements is at most', Ham_mat_part.nnz * size)
    comm.barrier()
    time1 = time.time()
    if rank == 0:
        print(f'Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(f'Start to solve...')
        energy = sort(eigvalsh(Ham_mat_part.toarray()))
    return energy[:num]

def arnoldi(indb_subset,basis,invbasis,s,num):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    if rank==0:
        print('Nb. of Non-zero elements is at most', Ham_mat_part.nnz * size)
    comm.barrier()
    time1 = time.time()
    if rank == 0:
        print(f'Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(f'Start to solve...')
        energy = sort(eigsh(Ham_mat_part,k=num,which='SA',return_eigenvectors=False))
    return energy[:num]


def lanczos_mpi(indb_subset,basis,invbasis,s,num,numiteration):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    nnz_part = Ham_mat_part.nnz
    ########################### MPI ###########################
    nnz_all = comm.allgather(nnz_part)
    nnz_max = amax(nnz_all)
    nnz_min = amin(nnz_all)
    nnz_tot = sum(nnz_all)
    ###########################################################
    if rank==0:
        print(f'Nb. of Non-zero elements is {nnz_tot} in total and it ranges from {nnz_min} to {nnz_max} in each process.')
    comm.barrier()
    time1 = time.time()
    if rank == 0:
        print(f'Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(f'Start to generate iteratively Lanczos matrix...') 
    # Lanczos
    a, b = [], []
    ########################### MPI ###########################
    if rank == 0:
        u2 = (2.0*np.random.random_sample(basis_size)+1.0) + 1.j * (2.0*np.random.random_sample(basis_size)+1.0)
        u2 = u2/norm(u2)
    else:
        u2 = None
    u2 = comm.bcast(u2, root=0)
    ###########################################################
    for iter in range(numiteration):
        time10 = time.time()
        if rank == 0:
            if (iter+1)%10 == 0 or iter==0:
                print(f'{iter+1}/{numiteration}', end='\t')
        u2t_part = Ham_mat_part.dot(u2)
        u2_part = u2[indb_subset]
        u2_measure_part = vdot(u2_part,u2t_part)
        ########################### MPI ###########################
        u2t_allpart = comm.allgather(u2t_part)
        u2t = concatenate(u2t_allpart) # H|u2>
        u2_measure = comm.allreduce(u2_measure_part,op=MPI.SUM)
        ###########################################################
        a.append(u2_measure)
        u3 = u2t - a[iter]*u2
        if iter>0: 
            u3 -= b[iter-1]*u1
        b.append(norm(u3))
        u3 /= b[iter]
        u1 = u2
        u2 = u3
        time11 = time.time()
        if rank == 0:
            if (iter+1)%10 == 0 or iter==0:
                print(f'{(time11-time10):.2f}s')
    time2 = time.time()
    if rank == 0:
        print(f'Lanczos matrix is generated for {(time2-time1):.2f}s.')
    tridiag = diag(a)
    tridiag+= diag(b[:-1],k=-1)
    energy = eigvalsh(tridiag)
    
    return energy[:num]

def lanczos_partial_reortho_mpi(indb_subset,basis,invbasis,s,num,numiteration,ortho_tol=1e-6):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    del row,col,data
    nnz_part = Ham_mat_part.nnz
    ########################### MPI ###########################
    nnz_all = comm.allgather(nnz_part)
    nnz_max = amax(nnz_all)
    nnz_min = amin(nnz_all)
    nnz_tot = sum(nnz_all)
    ###########################################################
    if rank==0:
        print(f'Nb. of Non-zero elements is {nnz_tot} in total and it ranges from {nnz_min} to {nnz_max} in each process.')
    comm.barrier()
    time1 = time.time()
    if rank == 0:
        print(f'Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(f'Start to generate iteratively Lanczos matrix...') 
    # Lanczos
    a, b = [], []
    Q = empty((0,length_subset),'complex')

    ########################### MPI ###########################
    if rank == 0:
        u2 = (2.0*np.random.random_sample(basis_size)+1.0) + 1.j * (2.0*np.random.random_sample(basis_size)+1.0)
        u2 = u2/norm(u2)
    else:
        u2 = None
    u2 = comm.bcast(u2, root=0)
    ###########################################################
    for iter in range(numiteration):
        time10 = time.time()
        if rank == 0:
            if (iter+1)%10 == 0 or iter==0:
                print(f'{iter+1}/{numiteration}', end='\t')
        u2t_part = Ham_mat_part.dot(u2)
        u2_part = u2[indb_subset]
        u2_measure_part = vdot(u2_part,u2t_part)
        ########################### MPI ###########################
        u2t_allpart = comm.allgather(u2t_part)
        u2t = concatenate(u2t_allpart) # H|u2>
        u2_measure = comm.allreduce(u2_measure_part,op=MPI.SUM)
        ###########################################################
        a.append(u2_measure.real)
        u3 = u2t - a[iter]*u2
        if iter>0: 
            u3 -= b[iter-1]*u1
        for i in range(Q.shape[0]):
            q_part = Q[i]
            overlap_part = vdot(q_part,u3[indb_subset])
            ########################### MPI ###########################
            q_allpart = comm.allgather(q_part)
            q = concatenate(q_allpart)
            overlap = comm.allreduce(overlap_part,op=MPI.SUM)
            ###########################################################
            if abs(overlap) > ortho_tol:
                u3 -= overlap * q
        b_next = norm(u3)
        if b_next < 1e-12:
            print("Lanczos terminated early at step", i)
            break
        b.append(b_next)
        u1 = u2
        u2 = u3 / b_next
        Q = vstack((Q,u2[indb_subset]))
        time11 = time.time()
        if rank == 0:
            if (iter+1)%10 == 0 or iter==0:
                print(f'{(time11-time10):.2f}s')
    time2 = time.time()
    if rank == 0:
        print(f'Lanczos matrix is generated for {(time2-time1):.2f}s.')
    tridiag = diag(a)
    tridiag+= diag(b[:-1],k=-1)
    energy = eigvalsh(tridiag)
    
    return energy[:num]


def lanczos_partial_reortho_CauchyCV_mpi(indb_subset,basis,invbasis,s,num,numiteration=200,cauchy_tol=1e-6,ortho_tol=1e-6):
    time0 = time.time()
    def lanczosSpectrum(a,b,num):
        tridiag = diag(a)
        tridiag+= diag(b,k=-1)
        return eigvalsh(tridiag)[:num]
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    nnz_part = Ham_mat_part.nnz
    ########################### MPI ###########################
    nnz_all = comm.allgather(nnz_part)
    nnz_max = amax(nnz_all)
    nnz_min = amin(nnz_all)
    nnz_tot = sum(nnz_all)
    ###########################################################
    if rank==0:
        print(f'Nb. of Non-zero elements is {nnz_tot} in total and it ranges from {nnz_min} to {nnz_max} in each process.')
    comm.barrier()
    time1 = time.time()
    if rank == 0:
        print(f'Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(f'Start to generate iteratively Lanczos matrix with Cauchy tol. = {cauchy_tol}...') 
    # Lanczos
    a, b = [], []
    Q = empty((0,length_subset),'complex')
    ########################### MPI ###########################
    if rank == 0:
        u2 = (2.0*np.random.random_sample(basis_size)+1.0) + 1.j * (2.0*np.random.random_sample(basis_size)+1.0)
        u2 = u2/norm(u2)
    else:
        u2 = None
    u2 = comm.bcast(u2, root=0)
    ###########################################################
    for iter in range(numiteration):
        time10 = time.time()
        if rank == 0:
            if iter>num:
                print(f'{iter+1}/{numiteration}', end='\t')
        u2t_part = Ham_mat_part.dot(u2)
        u2_part = u2[indb_subset]
        u2_measure_part = vdot(u2_part,u2t_part)
        ########################### MPI ###########################
        u2t_allpart = comm.allgather(u2t_part)
        u2t = concatenate(u2t_allpart) # H|u2>
        u2_measure = comm.allreduce(u2_measure_part,op=MPI.SUM)
        ###########################################################
        a.append(u2_measure.real)
        u3 = u2t - a[iter]*u2
        if iter>0: 
            u3 -= b[iter-1]*u1
        for i in range(Q.shape[0]):
            q_part = Q[i]
            overlap_part = vdot(q_part,u3[indb_subset])
            ########################### MPI ###########################
            q_allpart = comm.allgather(q_part)
            q = concatenate(q_allpart)
            overlap = comm.allreduce(overlap_part,op=MPI.SUM)
            ###########################################################
            if abs(overlap) > ortho_tol:
                u3 -= overlap * q
        b_next = norm(u3)
        if b_next < 1e-12:
            print("Lanczos terminated early at step", i)
            break
        b.append(b_next)
        u1 = u2
        u2 = u3 / b_next
        Q = vstack((Q,u2[indb_subset]))
        time11 = time.time()
        if iter == num:
            energy_old = lanczosSpectrum(a,b[:-1],num)
        if iter > num:
            energy_new = lanczosSpectrum(a,b[:-1],num)
            dE = norm(energy_new-energy_old)
            if rank == 0:
                print(f'{(time11-time10):.2f}s', end='\t')
                print(' '.join([f'{x:.16f}' for x in energy_new]), end='\t')
                print(dE)
            energy_old = energy_new
            if dE < cauchy_tol:
                time2 = time.time()
                energy_final = energy_new
                if rank == 0:
                    print(f'Convergence is reached for {(time2-time1):.2f}s.')
                break
        if iter==numiteration-1:
            energy_final = energy_new
            time2 = time.time()
            if rank == 0:
                print(f'WARNING: Convergence is NOT reached and the error is {dE}!!!')
                print(f'Time for all the iterations is {(time2-time1):.2f}s.')
        

    return energy_final

def lanczos_partial_reortho_RitzCV_mpi(indb_subset,basis,invbasis,s,num,numiteration,res_tol=1e-6,ortho_tol=1e-6):
    time0 = time.time()
    def compute_residuals(a, b):
        T = diag(a) + diag(b, k=-1) + diag(b, k=1)
        eigvals, eigvecs = eigh(T)
        residuals = absolute(b[-1]) * absolute(eigvecs[-1, :])  # 只需最后一项
        return eigvals, residuals
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,s,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    nnz_part = Ham_mat_part.nnz
    ########################### MPI ###########################
    nnz_all = comm.allgather(nnz_part)
    nnz_max = amax(nnz_all)
    nnz_min = amin(nnz_all)
    nnz_tot = sum(nnz_all)
    ###########################################################
    if rank==0:
        print(f'Nb. of Non-zero elements is {nnz_tot} in total and it ranges from {nnz_min} to {nnz_max} in each process.')
    comm.barrier()
    time1 = time.time()
    if rank == 0:
        print(f'Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(f'Start to generate iteratively Lanczos matrix with residual tol. = {res_tol}...') 
    # Lanczos
    a, b = [], []
    Q = empty((0,length_subset),'complex')
    ########################### MPI ###########################
    if rank == 0:
        u2 = (2.0*np.random.random_sample(basis_size)+1.0) + 1.j * (2.0*np.random.random_sample(basis_size)+1.0)
        u2 = u2/norm(u2)
    else:
        u2 = None
    u2 = comm.bcast(u2, root=0)
    ###########################################################
    for iter in range(numiteration):
        time10 = time.time()
        if rank == 0:
            if iter>=num:
                print(f'{iter+1}/{numiteration}', end='\t')
        u2t_part = Ham_mat_part.dot(u2)
        u2_part = u2[indb_subset]
        u2_measure_part = vdot(u2_part,u2t_part)
        ########################### MPI ###########################
        u2t_allpart = comm.allgather(u2t_part)
        u2t = concatenate(u2t_allpart) # H|u2>
        u2_measure = comm.allreduce(u2_measure_part,op=MPI.SUM)
        ###########################################################
        a.append(u2_measure.real)
        u3 = u2t - a[iter]*u2
        if iter>0: 
            u3 -= b[iter-1]*u1
        for i in range(Q.shape[0]):
            q_part = Q[i]
            overlap_part = vdot(q_part,u3[indb_subset])
            ########################### MPI ###########################
            q_allpart = comm.allgather(q_part)
            q = concatenate(q_allpart)
            overlap = comm.allreduce(overlap_part,op=MPI.SUM)
            ###########################################################
            if abs(overlap) > ortho_tol:
                u3 -= overlap * q
        b_next = norm(u3)
        if b_next < 1e-12:
            print("Lanczos terminated early at step", i)
            break
        b.append(b_next)
        u1 = u2
        u2 = u3 / b_next
        Q = vstack((Q,u2[indb_subset]))
        time11 = time.time()

        if iter>= num:
            ritz_vals, residuals = compute_residuals(a, b[:-1])
            top_k_idx = argsort(ritz_vals)[:num]
            top_residuals = residuals[top_k_idx]
            ritz_res = amax(top_residuals)
            if rank == 0:
                print(f'{(time11-time10):.2f}s', end='\t')
                print(' '.join([f'{x:.16f}' for x in ritz_vals[top_k_idx]]), end='\t')
                print(ritz_res)
            if ritz_res < res_tol:
                time2 = time.time()
                energy_final = ritz_vals[top_k_idx]
                if rank == 0:
                    print(f'Convergence is reached for {(time2-time1):.2f}s.')
                break
        if iter==numiteration-1:
            energy_final = ritz_vals[top_k_idx]
            time2 = time.time()
            if rank == 0:
                print(f'WARNING: Convergence is NOT reached and the max Ritz residual value is {ritz_res}!!!')
                print(f'Time for all the iterations is {(time2-time1):.2f}s.')
    return energy_final

if __name__=='__main__':
    time0 = time.time()
    
    if rank==0:
        if size>1 and algo<0:
            print('')
            print('ERROR: Full matrix and Arnoldi algo. is only single-threaded!!!')
            print('')
            sys.exit(0)

        print('+==================================================================================+')
        print(f'+                        Running in parallel on {size:4d} CPUs                          ')
        print(f'+              nu={p}/{q}, Nphi={Nphi}, Ne={Ne}, ratio a/b={ratio_rect:.2f}       ')
        print('+==================================================================================+')

       

    Vc_data,j_Vc = generate_Vc()
    Coulomb_mat = zeros((Nphi,Nphi,Nphi,Nphi), 'complex')

    for i in range(Vc_data.shape[0]):
        j1,j2,j3,j4 = j_Vc[i]
        matele = Vc_data[i]
        Coulomb_mat[j1,j2,j3,j4] = matele

    time1 = time.time()
    if rank == 0:
        print(f'Data for Hamiltonian matrix elements are generated for {(time1-time0):.2f}s.')
        print(' ')
        print('+==================================================================================+')
        print(f'+                        Algorithm: {algostr}                         ')
        print('+                       Generate Matrix elements and Lanczos-solve Ham.          ')
        print(f'+                        keeping the {num} lowest energies with Niter={numiteration}                       ')
        print('+==================================================================================+')

    en_lanczos = zeros((N*N,num),'float')
    for st in range(N*N): # s*N+t
        s_select,t_select = divmod(st,N)
        if rank == 0:
            print(f'--------------------Block (s,t)=({s_select},{t_select})---------------------------')
        time20 = time.time()
        
        basis = create_Hilbert(s_select,t_select) # [(rep0,R0,Q0),(rep1,R1,Q1),...] given index, return rep Ra Qat
        time21 = time.time()
        if rank == 0:
            print(f'* Hilbert space basis are generated for {(time21-time20):.2f}s!')
        
        basis_size = basis.shape[0]
        basis = ascontiguousarray(basis)

        invbasis = Dict.empty(key_type = types.int64, value_type = types.int64) # given rep, return index and Ra
        for indc, doublet in enumerate(basis):
            repa = doublet[0]
            Ra = doublet[1]
            invbasis[repa] = indc
        ########################### MPI ###########################
        # If the size of array to be parallelized *can not be divided* by the number of cores,
        # the array will be diveded into subsets with 2 types of size:
        # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
        subset_size,num_more=divmod(basis_size,size)
        indb_subsets=[range(basis_size)[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else range(basis_size)[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] 
        indb_subset=comm.scatter(indb_subsets,root=0) # divide basis into pieces (b in <b|V|b'>) and scatter them
        indb_subset = ascontiguousarray(indb_subset, dtype=np.int64)
        ###########################################################
        if algo == -2:
            en_lanczos[st] = fullmatrix(indb_subset,basis,invbasis,s_select,num)
        if algo == -1:
            en_lanczos[st] = arnoldi(indb_subset,basis,invbasis,s_select,num)
        if algo == 0:
            en_lanczos[st] = lanczos_mpi(indb_subset,basis,invbasis,s_select,num,numiteration)
        elif algo == 1:
            en_lanczos[st] = lanczos_partial_reortho_mpi(indb_subset,basis,invbasis,s_select,num,numiteration,ortho_tol=1e-6)
        elif algo == 2:
            en_lanczos[st] = lanczos_partial_reortho_CauchyCV_mpi(indb_subset,basis,invbasis,s_select,num,numiteration,cauchy_tol=1e-12,ortho_tol=1e-6)
        elif algo == 3:
            en_lanczos[st] = lanczos_partial_reortho_RitzCV_mpi(indb_subset,basis,invbasis,s_select,num,numiteration,res_tol=1e-12,ortho_tol=1e-6)
            

        time22 = time.time()
        if rank == 0:
            print(f'Lanczos spectrum for Block (s,t)=({s_select},{t_select}) is solved for {(time22-time21):.2f}s.')
            print(f' ')
            
    time3 = time.time()
    if rank == 0:

        fig,ax=plt.subplots(figsize=(4,4))
        Emin = amin(en_lanczos)
        knorm_list = zeros((N*N),'float')

        if p*q*(Ne-1) % 2 == 0:
            for st in range(N*N): # s*N+t
                s_select,t_select = divmod(st,N)
                if s_select>N//2:
                    s_select = s_select - N
                if t_select>N//2:
                    t_select = t_select - N
                kvec = sqrt(2*pi/(Nphi*ratio_rect)) * array([s_select-s0,ratio_rect*(t_select-t0)])
                knorm_list[st] = norm(kvec)
        elif p*q*(Ne-1) % 2 == 1:
            for st in range(N*N): # s*N+t
                s_select,t_select = divmod(st,N)
                kvec = sqrt(2*pi/(Nphi*ratio_rect)) * array([s_select-s0,ratio_rect*(t_select-t0)])
                knorm_list[st] = norm(kvec)

        
        for i in range(num):
            ax.scatter(knorm_list, en_lanczos[:,i]-Emin, color='r', marker='o')
        ax.set_ylabel(r'$E-E_{min}$ ($e^2/\epsilon l_B$)')
        ax.set_xlabel(rf'$k l_B$')
        knorm_max = amax(knorm_list)
        ax.set_xlim(0-0.1*sqrt(2*pi/(Nphi*ratio_rect)), knorm_max+0.1*sqrt(2*pi/(Nphi*ratio_rect)))

        fig.tight_layout()
        fig.savefig(f'ED_spec_Nphi{Nphi}_Ne{Ne}_ratio{ratio_rect}_algo{algo}_Niter{numiteration}_Nproc{size}.pdf')

        print('--------------------------------------------------')
        print(f'* Total time elapse: {time3-time0}s.')



