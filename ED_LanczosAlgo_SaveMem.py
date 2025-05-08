# -*- coding: utf-8 -*-
"""
Created on Mon Apr 7 09:23:20 2025

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



K0 = int(sys.argv[1])
K1 = int(sys.argv[2])
Ne = int(sys.argv[3])
numiteration = int(sys.argv[4])
num = 5
algo = int(sys.argv[5])

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


L = K1*K0 # Given mesh K0,K1
hilbertsize = int(binom(L,Ne))
basis_rank = {}

@njit
def totmomentum(c):
    "calculate total momentum of a conf integer with given (K0,K1) mesh of length L=K0*K1"
    K0tot = 0
    K1tot = 0
    for ik0 in range(K0):
        for ik1 in range(K1):
            iK = ik0*K1 + ik1
            if c & (1 << iK):
                K0tot += ik0
                K1tot += ik1
    return [K0tot%K0,K1tot%K1]


def fillhilbert(conf,Ktot): 
    global basis_rank
    K0c,K1c = totmomentum(conf)
    Ktotc = K0c*K1+K1c
    if Ktotc == Ktot:
        basis_rank.append(conf)


def create_Hilbert(Ktot):
    timeh0 = time.time()
    global basis_rank
    # basis_rank = array([]) 
    basis_rank = [] 
    K0t,K1t = divmod(Ktot,K1)

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

    LL = L-L//2
    LR = L//2
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
            indc_subset=comm.scatter(indc_subsets,root=0)
            ###########################################################
            for indc in indc_subset:
                indcL = indc // NconfR
                indcR = indc % NconfR
                confL = leftConfs[kl][indcL]
                confR = rightConfs[kr][indcR]
                conf = (confL<<LR)^confR
                fillhilbert(conf,Ktot)
    comm.barrier()
    timeh1 = time.time()
    if rank == 0:
        print(f'Ktot=({K0t},{K1t}): confs are found for {(timeh1-timeh0):.2f}s.')
    ########################### MPI ###########################
    basis_rank_gather = comm.allgather(basis_rank)
    basis_merged = concatenate(basis_rank_gather)
    ###########################################################
    timeh2 = time.time()
    if rank == 0:
        print(f'Basis are merged for {(timeh2-timeh1):.2f}s.')
    if rank == 0:
        print(f"{len(basis_merged)} over {hilbertsize} configurations")
    del basis_rank
    return basis_merged.astype('int')


@njit
def Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat):
    klist = []
    for k1 in range(K0*K1):
        for k2 in range(k1+1,K0*K1):
            for k3 in range(K0*K1):
                for k4 in range(k3+1,K0*K1):
                    k1v = (k1//K1,k1%K1)
                    k2v = (k2//K1,k2%K1)
                    k3v = (k3//K1,k3%K1)
                    k4v = (k4//K1,k4%K1)
                    momentum_consv1 = (k1v[0]+k2v[0]-k3v[0]-k4v[0]) % K0
                    momentum_consv2 = (k1v[1]+k2v[1]-k3v[1]-k4v[1]) % K1
                    # Momentum conservation actually ensures that k1,k2,k3,k4 are not distinct from each other.
                    if momentum_consv1==0 and momentum_consv2==0:
                        klist.append([k1,k2,k3,k4])
    
    klist = array(klist)
    row, col, data = [], [], []
    subset_size = len(indb_subset)
    for nconf, indc in enumerate(indb_subset):
        if rank == 0 and (nconf==0 or (nconf+1)%1e5==0):
            print(nconf+1,'/',subset_size)
        conf = basis[indc]
        for ik in range(klist.shape[0]):
            k1,k2,k3,k4 = klist[ik]
            sign = 1.
            if occupation(conf,k3) & occupation(conf,k4):
                # k4
                sign *= (-1)**count_ones_from(conf,k4+1)
                newconf = conf & (~(1 << k4))
                # k3
                sign *= (-1)**count_ones_from(newconf,k3+1)
                newconf = newconf & (~(1 << k3))
                if (1-occupation(newconf,k1)) & (1-occupation(newconf,k2)):
                    # k2
                    sign *= (-1)**count_ones_from(newconf,k2+1)
                    newconf = newconf | (1 << k2)
                    # k1
                    sign *= (-1)**count_ones_from(newconf,k1+1)
                    newconf = newconf | (1 << k1)
                    Vq = Coulomb_mat[k1,k2,k3,k4] - Coulomb_mat[k2,k1,k3,k4] - Coulomb_mat[k1,k2,k4,k3] + Coulomb_mat[k2,k1,k4,k3]
                    rescoef = conj(Vq*sign) # ATT: what we need is <conf|V|newconf>
                    if absolute(rescoef)>1e-14:
                        indc_new = invbasis[newconf]
                        row.append(nconf)
                        col.append(indc_new)
                        data.append(rescoef)
    return row,col,data


@njit
def Ek_mat_part(indb_subset,basis,Ek):
    row, col, data = [], [], []
    for nc, indc in enumerate(indb_subset):
        conf = basis[indc]
        Ekt = 0.
        for ik in range(L):
            if occupation(conf,ik):
                Ekt += Ek[ik]
        row.append(nc)
        col.append(indc)
        data.append(Ekt)
    return row,col,data

def fullmatrix(indb_subset,basis,invbasis,num):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    row,col,data = Ek_mat_part(indb_subset,basis,Ek)
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

def arnoldi(indb_subset,basis,invbasis,num):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    row,col,data = Ek_mat_part(indb_subset,basis,Ek)
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


def lanczos_mpi(indb_subset,basis,invbasis,num,numiteration):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    row,col,data = Ek_mat_part(indb_subset,basis,Ek)
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

def lanczos_partial_reortho_mpi(indb_subset,basis,invbasis,num,numiteration,ortho_tol=1e-6):
    time0 = time.time()
    # Generate matrix element
    if rank==0:
        print('Start to compute Hamiltonian matrix elements...')
    length_subset = len(indb_subset)
    Ham_mat_part = csr_matrix((length_subset,basis_size),dtype='complex')
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    row,col,data = Ek_mat_part(indb_subset,basis,Ek)
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


def lanczos_partial_reortho_CauchyCV_mpi(indb_subset,basis,invbasis,num,numiteration=200,cauchy_tol=1e-6,ortho_tol=1e-6):
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
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    row,col,data = Ek_mat_part(indb_subset,basis,Ek)
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

def lanczos_partial_reortho_RitzCV_mpi(indb_subset,basis,invbasis,num,numiteration,res_tol=1e-6,ortho_tol=1e-6):
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
    row,col,data = Vc_mat_part(indb_subset,basis,invbasis,K0,K1,Coulomb_mat)
    Ham_mat_part += csr_matrix((data,(row,col)),shape=(length_subset,basis_size),dtype='complex')
    row,col,data = Ek_mat_part(indb_subset,basis,Ek)
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
        print('+==================================================================================+')
        print(f'+                        Running in parallel on {size:4d} CPUs                          ')
        print(f'+                               K mesh=({K0},{K1}), Ne={Ne}                               ')
        print('+==================================================================================+')

    # Read data
    Vc_data = load(f'Ne{Ne}_matrix_element.npz')['Vc']
    k_Vc = load(f'Ne{Ne}_matrix_element.npz')['klist']
    Ek_data = load(f'Ne{Ne}_one_body_term.npz')['Ek']
    k_Ek = load(f'Ne{Ne}_one_body_term.npz')['klist']

    Ek = zeros(L,'float')
    Coulomb_mat = zeros((L,L,L,L), 'complex')

    for i in range(Vc_data.shape[0]):
        ikx1,iky1,ikx2,iky2,ikx3,iky3,ikx4,iky4 = k_Vc[i]
        matele = 0.5*Vc_data[i]
        ik1 = ikx1*K1 + iky1
        ik2 = ikx2*K1 + iky2
        ik3 = ikx3*K1 + iky3
        ik4 = ikx4*K1 + iky4
        Coulomb_mat[ik1,ik2,ik3,ik4] = matele

    for i in range(Ek_data.shape[0]):
        ikx,iky= k_Ek[i]
        matele = Ek_data[i]
        ik = ikx*K1+ iky
        Ek[ik] = matele

    time1 = time.time()
    if rank == 0:
        print(f'Data for Hamiltonian matrix elements are loaded for {(time1-time0):.2f}s.')
        print(' ')
        print('+==================================================================================+')
        print(f'+                        Algorithm: {algostr}                         ')
        print('+                       Generate Matrix elements and Lanczos-solve Ham.          ')
        print(f'+                        keeping the {num} lowest energies with Niter={numiteration}                       ')
        print('+==================================================================================+')
    en_lanczos = zeros((L,num),'float')
    for Ktot in range(L):
        K0t,K1t = divmod(Ktot,K1)
        if rank == 0:
            print(f'--------------------Block ({K0t},{K1t})---------------------------')
        time20 = time.time()
        
        basis = create_Hilbert(Ktot)
        time21 = time.time()
        if rank == 0:
            print(f'* Hilbert space basis are generated for {(time1-time0):.2f}s!')
        
        basis_size = len(basis)
        basis = ascontiguousarray(basis)
        invbasis = Dict.empty(key_type = types.int64, value_type = types.int64)
        for key, value in enumerate(basis):
            invbasis[value] = key
        ########################### MPI ###########################
        # If the size of array to be parallelized *can not be divided* by the number of cores,
        # the array will be diveded into subsets with 2 types of size:
        # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
        subset_size,num_more=divmod(basis_size,size)
        indb_subsets=[range(basis_size)[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else range(basis_size)[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] 
        indb_subset=comm.scatter(indb_subsets,root=0)
        indb_subset = ascontiguousarray(indb_subset)
        ###########################################################
        if algo == -2:
            en_lanczos[Ktot] = fullmatrix(indb_subset,basis,invbasis,num)
        if algo == -1:
            en_lanczos[Ktot] = arnoldi(indb_subset,basis,invbasis,num)
        if algo == 0:
            en_lanczos[Ktot] = lanczos_mpi(indb_subset,basis,invbasis,num,numiteration)
        elif algo == 1:
            en_lanczos[Ktot] = lanczos_partial_reortho_mpi(indb_subset,basis,invbasis,num,numiteration,ortho_tol=1e-6)
        elif algo == 2:
            en_lanczos[Ktot] = lanczos_partial_reortho_CauchyCV_mpi(indb_subset,basis,invbasis,num,numiteration,cauchy_tol=1e-12,ortho_tol=1e-6)
        elif algo == 3:
            en_lanczos[Ktot] = lanczos_partial_reortho_RitzCV_mpi(indb_subset,basis,invbasis,num,numiteration,res_tol=1e-12,ortho_tol=1e-6)
            

        time22 = time.time()
        if rank == 0:
            print(f'Lanczos spectrum for Block ({K0t},{K1t}) is solved for {(time22-time21):.2f}s.')
            print(f' ')
            
    time3 = time.time()
    if rank == 0:
        fig,ax=plt.subplots(figsize=(4,4))
        Emin =amin(en_lanczos)
        list_K = arange(L)
        for i in range(num):
            ax.plot(list_K,(en_lanczos[:,i]-Emin) * 1000, 'ro')
        ax.set_ylabel(r'$E-E_{min}$ (meV)')
        ax.set_xlabel(rf'$k_0 \times {K1} + k_1$')
        ax.set_xticks(arange(0,L,3))
        fig.tight_layout()
        fig.savefig(f'EDspec_algo{algo}_Niter{numiteration}_Ne{Ne}_mesh{K0}and{K1}_Nproc{size}.pdf')

        print('--------------------------------------------------')
        print(f'* Total time elapse: {time3-time0}s.')


