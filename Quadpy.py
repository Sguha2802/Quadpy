import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
from tabulate import tabulate
from scipy.linalg import sqrtm
import time    
import seaborn as sns
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor, thermal_dm, anim_matrix_histogram,
                   anim_fock_distribution)

#Function to define 2d X 2d Omega
def Omega(d):
    om = np.zeros((2*d,2*d),dtype=np.complex_)
    i = 0
    while i<d:
        om[i][i] = 1
        i += 1
    while i<2*d:
        om[i][i] = -1
        i += 1
    return om


def PrintMatrix(X,n=4):
    row, col = X.shape
    Y = np.zeros((row, col), dtype=object)
    for i in range(0,row):
        for j in range(0,col):
            temp = np.round(X.real[i][j],n) + 1j*np.round(X.imag[i][j],n)
            Y[i][j] = str(temp)
    print(tabulate(Y))  

def PrintList(X,n=4):
    row = len(X)
    Y = np.zeros((row,1), dtype=object)
    
    for i in range(0,row):
        temp = np.round(X.real[i],n) + 1j*np.round(X.imag[i],n)
        Y[i][0] = str(temp)
    print(tabulate(Y))  

def is_hermitian(x, eps=1.e-8):
    return np.all(abs(np.conjugate(np.transpose(x))-x)<eps)

def is_transpose(x, eps=1.e-8):
    return np.all(abs(np.transpose(x)-x)<eps)

#Function to check if a transformation is Boguliobov
def check_Bogoliubov(T,eps=1.e-12):
    N = int(len(T)/2)
    k = 0
    for i in range(N):
        uv = T[:,i]
        v_u_ = T[:,i+N]
        u = uv[0:N]
        v = uv[N:2*N]
        v_ = v_u_[0:N]
        u_ = v_u_[N:2*N]
        if np.all(np.imag(np.conjugate(u)-u_) > eps):
            k += 1
        if np.all(np.real(np.conjugate(u)-u_) > eps):
            k += 1
        if np.all(np.imag(np.conjugate(v)-v_) > eps):
            k += 1
        if np.all(np.real(np.conjugate(v)-v_) > eps):
            k += 1
    return (k == 0)

def H_bar(H):
    N = int(len(H)/2)
    om = Omega(N)
    H_half = sqrtm(H)
    H_ = H_half @ om @ H_half
    
    return H_

def U_D(H):
    H_ = np.array(H_bar(H),dtype=np.complex_)
    eig_vals, eig_vecs = np.linalg.eig(H_)
    eigs = eig_vals
    N = int(len(eig_vals)/2)
    U = np.zeros((2*N,2*N),dtype=np.complex_)
    indices = np.argsort(eig_vals)
    neg_indices = indices[0:N]
    pos_indices = indices[N:2*N]
    final_indices = np.concatenate((pos_indices,np.flip(neg_indices)))
    D = eigs[final_indices]
    U = eig_vecs[:,final_indices]
    return (U,D)

def Bogoliubov_Transform(H):
    N = int(len(H)/2)
    H_ = H_bar(H)
    sqrt_H = sqrtm(H)
    U, diag_elems = U_D(H)
    U_dag = np.conjugate(np.transpose(U))
    D = np.diag(diag_elems[0:N])
    sqrt_D = np.sqrt(D)
    inv_sqrt_D = np.linalg.inv(sqrt_D)
    D_bar = np.kron(np.identity(2), inv_sqrt_D)
    T = D_bar @ U_dag @ sqrt_H
    return [T,U,diag_elems]

def Zetad_Zeta_Matrix(T,etd_et):
    T_conj = np.conjugate(T)
    T_trans = np.transpose(T)
    return T_conj @ etd_et @ T_trans

def mod_zeta_corr(T_inv,Zd_Z_matrix,p,q):

    A = T_inv
    A_dag = np.transpose(np.conjugate(T_inv))

    m_zeta_corr0 = Zd_Z_matrix
        
    m_zeta_corr1 = np.transpose(np.multiply(A_dag[:,p],np.transpose(m_zeta_corr0)))
    m_zeta_corr = np.multiply(A[q,:],m_zeta_corr1)
    
    return m_zeta_corr

def exp_i(x):
    two_pi = 2*np.pi
    return np.exp(1j*(x%two_pi))

def exp_ilambda_t(lam,t):
    
    e_ilam_t = np.exp(1j*lam*t)
    return e_ilam_t

def Eta_correlation_t(lam,mzeta_corr_pq,t):
    e_ilam_t = exp_ilambda_t(lam,t)
    etd_et_t = e_ilam_t @ mzeta_corr_pq @ np.conjugate(np.transpose(e_ilam_t))
    
    return etd_et_t

def Evolve(lam,mzeta_corr_pq,t_max,del_t = 0.01):
    Time = np.linspace(0,t_max,int((t_max/del_t)),endpoint=True)
    Etd_Et_T = np.zeros(len(Time),dtype=np.complex_)
    for i in range(0, len(Time)):
        Etd_Et_T[i] = Eta_correlation_t(lam,mzeta_corr_pq,Time[i])
    return (Time,Etd_Et_T)

def TimeEvolution(H,Adag_A0,p,q,t_end,dt=0.01):
    T, U, λ = Bogoliubov_Transform(H)
    
    T = np.array(T,dtype=np.complex_)
    
    A = np.linalg.inv(T)
    
    zeta_d_zeta = Zetad_Zeta_Matrix(T,Adag_A0)
    
    modified_Zeta_Corr = mod_zeta_corr(A,zeta_d_zeta,p-1,q-1)
    
    initial_val_error = np.abs(Adag_A0[p-1][q-1]-Eta_correlation_t(λ,modified_Zeta_Corr,0))
    
    if initial_val_error > 1.e-13:
        print('Initial Value error: ',initial_val_error)

    Time, Corr = Evolve(λ,modified_Zeta_Corr,t_end,del_t = dt)

    return [Time, Corr]

def Init_Corr(N_b, ψ, N=100):
    a = []
    adag = []
    for i in range(N_b):
        ai_t = destroy(N)
        I_start = qeye(N)
        I_end = qeye(N)
        
        for j in range(i):
            if j == i-1:
                ai_t = tensor(I_start,ai_t)
            I_start = tensor(I_start,qeye(N))
            
        for k in range(N_b-i-1):
            if k == N_b-i-2:
                ai_t = tensor(ai_t,I_end)
            I_end = tensor(I_end,qeye(N))
            
        a.append(ai_t)
        adag.append(ai_t.dag())

    A = np.concatenate((np.array(a),np.array(adag)))
    Adag = np.concatenate((np.array(adag),np.array(a))) 
    Init = np.zeros((2*N_b,2*N_b), dtype=np.complex_)
    
    for i in range(2*N_b):
        for j in range(2*N_b):
            Init[i][j] = expect(Adag[i] * A[j], ψ)

    return Init

#Naive Dipole Hamiltonian
def Hamilton_Matrix_dG(n,omega,omega_0,x,g_0,cd_choice,spectrum='linear',J=0.2,RWA=0,strong=0,mode_offset=0,sm=1):
    n_osc = len(omega)
    sqrt_L = np.sqrt(omega_0*np.pi)
    N = n_osc + n

    pi = np.pi


    if spectrum == 'linear':
        k_modes = np.arange(mode_offset+1,n+mode_offset+1)

        omega_bath = k_modes/omega_0

    
    g_kj = np.zeros((n_osc,n),dtype=np.complex_)
    H_S = np.zeros((N,N),dtype=np.complex_)
    K = np.zeros((N,N),dtype=np.complex_)
    omega_tot = np.concatenate((omega,omega_bath))


    H_S = H_S + np.diag(omega_tot)

    if cd_choice == 'sinkx':
        for i in range(n_osc):
            g_kj[i] = g_0 * np.sin(omega_bath*x[i])
            if strong == 1:
                g_kj[i] = g_kj[i]* np.sqrt(omega_bath/omega[i])

    if cd_choice == 'sinkx+coskx':
        for j in range(n_osc):
            for k in range(n):
                if strong == 0:
                    if k%2 == 1:
                        g_kj[j][k] = g_0 * np.sin(omega_bath[k]*x[j])
                    if k%2 == 0:
                        g_kj[j][k] = g_0 * np.cos(omega_bath[k]*x[j])
                if strong == 1:
                    if k%2 == 1:
                        g_kj[j][k] = g_0 * np.sin(omega_bath[k]*x[j]) * np.sqrt(omega_bath[k]/omega[j])
                    if k%2 == 0:
                        g_kj[j][k] = g_0 * np.cos(omega_bath[k]*x[j]) * np.sqrt(omega_bath[k]/omega[j])

    H_S[0:n_osc,n_osc:N] = g_kj
    H_S[n_osc:N,0:n_osc] = np.conjugate(np.transpose(g_kj))

    K[0:n_osc,n_osc:N] = np.conjugate(g_kj)
    K[n_osc:N,0:n_osc] = np.conjugate(np.transpose(g_kj))

    Temp_zero_0 = np.zeros((N,N),dtype=np.complex_)
    if RWA == 0:
        System_Bath = np.concatenate((np.concatenate((H_S,np.conjugate(K))),np.concatenate((K,np.conjugate(H_S)))),axis=1)

    if RWA == 1:
        System_Bath = np.concatenate((np.concatenate((H_S,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(H_S)))),axis=1)


    h = System_Bath

    return h

def Hamilton_Matrix_CG(n,omega,omega_0,x,g_0,cd_choice,RWA=0,strong=0,mode_offset=0):

    n_osc = len(omega)
    sqrt_L = np.sqrt(omega_0*np.pi)
    N = n_osc + n

    pi = np.pi
    omega_bath = np.arange(mode_offset+1,n+mode_offset+1)/omega_0
    f_kj = np.zeros((n_osc,n),dtype=np.complex_)
    g_kj = np.zeros((n_osc,n),dtype=np.complex_)
    H_S = np.zeros((N,N),dtype=np.complex_)
    K = np.zeros((N,N),dtype=np.complex_)
    omega_tot = np.concatenate((omega,omega_bath))


    H_S = H_S + np.diag(omega_tot)

    if cd_choice == 'sinkx':
        for i in range(n_osc):
            f_kj[i] = np.sin(omega_bath*x[i])

    if cd_choice == 'sinkx+coskx':
        for j in range(n_osc):
            for k in range(n):
                if k%2 == 1:
                    f_kj[j][k] = np.sin(omega_bath[k]*x[j])
                if k%2 == 0:
                    f_kj[j][k] = np.cos(omega_bath[k]*x[j])
    
    for j in range(n_osc):
        if strong == 0:
            g_kj[j] = g_0[j] * f_kj[j]

        if strong == 1:
            g_kj[j] = g_0[j] * f_kj[j] * np.sqrt(omega[j]/omega_bath)
    
    H_S[0:n_osc,n_osc:N] = -1j * g_kj
    H_S[n_osc:N,0:n_osc] = +1j * np.conjugate(np.transpose(g_kj))

    K[0:n_osc,n_osc:N] = 1j * g_kj
    K[n_osc:N,0:n_osc] = 1j * np.transpose(g_kj)

    alpha = np.zeros((n,n),dtype=np.complex_)
    beta = np.zeros((n,n),dtype=np.complex_)

    for k in range(n):
        for k_ in range(n):

            alpha[k][k_] = np.transpose(np.conjugate(g_kj[:,k])) @ (g_kj[:,k_]/omega)
            beta[k][k_] = np.transpose(g_kj[:,k]) @ (g_kj[:,k_]/omega)

    Temp_zero_0 = np.zeros((N,N),dtype=np.complex_)
    Temp_zero_1 = np.zeros((n_osc,N),dtype=np.complex_)
    Temp_zero_2 = np.zeros((n,n_osc),dtype=np.complex_)

    α_tilde = np.concatenate((Temp_zero_1,np.concatenate((Temp_zero_2,alpha),axis=1)))
    β_tilde = np.concatenate((Temp_zero_1,np.concatenate((Temp_zero_2,beta),axis=1)))

    if RWA == 0:
        System_Bath = np.concatenate((np.concatenate((H_S,K)),np.concatenate((np.conjugate(K),np.conjugate(H_S)))),axis=1)

        Bath_Bath = np.concatenate((np.concatenate((α_tilde,β_tilde)),np.concatenate((np.conjugate(β_tilde),np.conjugate(α_tilde)))),axis=1)
    if RWA == 1:
        System_Bath = np.concatenate((np.concatenate((H_S,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(H_S)))),axis=1)

        Bath_Bath = np.concatenate((np.concatenate((α_tilde,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(α_tilde)))),axis=1)


    h = System_Bath + 2*Bath_Bath

    return [h, omega_bath, f_kj]

def Hamilton_Matrix_DG(n,omega,omega_0,x,g_0,cd_choice,spectrum='linear',RWA=0,strong=0,mode_offset=0,sm=1):
    n_osc = len(omega)
    sqrt_L = np.sqrt(omega_0*np.pi)
    N = n_osc + n

    pi = np.pi

    if spectrum == 'linear':
        k_modes = np.arange(mode_offset+1,n+mode_offset+1)

        omega_bath = k_modes/omega_0
#     print(g_0)
    f_kj = np.zeros((n_osc,n),dtype=np.complex_)
    g_kj = np.zeros((n_osc,n),dtype=np.complex_)
    H_S = np.zeros((N,N),dtype=np.complex_)
    K = np.zeros((N,N),dtype=np.complex_)
    B_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    A_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    E_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    D_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    omega_tot = np.concatenate((omega,omega_bath))

    H_S = H_S + np.diag(omega_tot)

    if cd_choice == 'single':
        for j in range(n_osc):
            f_kj[j][sm] =  1

    if cd_choice == 'sinkx':
        for i in range(n_osc):
            f_kj[i] = np.sin(omega_bath*x[i])

    if cd_choice == 'sinkx+coskx':
        for j in range(n_osc):
            for k in range(n):
                if k%2 == 1:
                    f_kj[j][k] = np.sin(omega_bath[k]*x[j])
                if k%2 == 0:
                    f_kj[j][k] = np.cos(omega_bath[k]*x[j])

    for j in range(n_osc):
        if strong == 0:
            g_kj[j] = g_0 * f_kj[j]

        if strong == 1:
            g_kj[j] = g_0 * f_kj[j] * np.sqrt(omega_bath/omega[j])


    H_S[0:n_osc,n_osc:N] = -1j * g_kj
    H_S[n_osc:N,0:n_osc] = +1j * np.conjugate(np.transpose(g_kj))


    K[0:n_osc,n_osc:N] = +1j * np.conjugate(g_kj)
    K[n_osc:N,0:n_osc] = +1j * np.conjugate(np.transpose(g_kj))

    for j in range(n_osc):
        for l in range(n_osc):
            B_jl[j][l] = np.imag((f_kj[j]/omega_bath) @ np.transpose(np.conjugate(f_kj[l])))

    B_squared = B_jl @ B_jl

    for j in range(n_osc):
        for l in range(n_osc):
            A_jl[j][l] = np.real(np.conjugate(f_kj[j]) @ np.transpose(f_kj[l])) - g_0**2 * B_squared[j][l]

    for j in range(n_osc):
        for l in range(n_osc):
            E_jl[j][l] = (1/np.sqrt(omega[j]*omega[l]))*(2*A_jl[j][l] + 1j * B_jl[j][l]*(omega[j]+omega[l]))

            D_jl[j][l] = (1/np.sqrt(omega[j]*omega[l]))*(2*A_jl[j][l] + 1j * B_jl[j][l]*(omega[j]-omega[l]))


    Temp_zero_0 = np.zeros((N,N),dtype=np.complex_)
    Temp_zero_1 = np.zeros((N,n),dtype=np.complex_)
    Temp_zero_2 = np.zeros((n,n_osc),dtype=np.complex_)

    E = np.concatenate((np.concatenate((E_jl,Temp_zero_2),axis=0),Temp_zero_1),axis=1)
    D = np.concatenate((np.concatenate((D_jl,Temp_zero_2),axis=0),Temp_zero_1),axis=1)

    if RWA == 0:
        System_Bath = np.concatenate((np.concatenate((H_S,np.conjugate(K))),np.concatenate((K,np.conjugate(H_S)))),axis=1)

        System_System = np.concatenate((np.concatenate((E,np.conjugate(D))),np.concatenate((D,np.conjugate(E)))),axis=1)
    if RWA == 1:
        System_Bath = np.concatenate((np.concatenate((H_S,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(H_S)))),axis=1)

        System_System = np.concatenate((np.concatenate((E,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(E)))),axis=1)

    h = System_Bath + g_0**2 * System_System


    return [h, omega_bath, f_kj]

def Hamilton_Matrix_CG_WG(n,omega,omega_0,x,g_0,cd_choice,omega_c=0,spectrum='TB',RWA=0,strong=0,mode_offset=0,HP=0.25):
    n_osc = len(omega)
    L = omega_0*np.pi
    sqrt_L = np.sqrt(omega_0*np.pi)
    N = n_osc + n

    pi = np.pi

    if spectrum == 'linear':
        k_modes = 2*np.arange(mode_offset+1,n+mode_offset+1)/omega_0

        omega_bath = k_modes

    if spectrum == 'TB':
        k_modes_p = np.arange(0,(n*pi)/L,(2*pi)/L)
        k_modes_m = -np.flip(np.arange((2*pi)/L,(n*pi)/L+(2*pi)/L,(2*pi)/L))
        if omega_c == 0:
            omega_c = omega[0]
        omega_bath_p = omega_c - HP * np.cos(k_modes_p*(L/n))
        omega_bath_m = omega_c - HP * np.cos(k_modes_m*(L/n))

    f1_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)
    f2_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)
    g1_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)
    g2_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)

    H_S = np.zeros((N,N),dtype=np.complex_)
    K = np.zeros((N,N),dtype=np.complex_)

    omega_tot = np.concatenate((omega,omega_bath_m,omega_bath_p))
    
    H_S = H_S + np.diag(omega_tot)

    if cd_choice == 'dual_exp+-ikx':
        for i in range(n_osc):
            f1_kj[i] = np.exp(+1j * k_modes_m * x[i])
            f2_kj[i] = np.exp(+1j * k_modes_p * x[i])

    for j in range(n_osc):
        if strong == 0:
            g1_kj[j] = g_0 * f1_kj[j]
            g2_kj[j] = g_0 * f2_kj[j]

        if strong == 1:
            g1_kj[j] = g_0 * f1_kj[j] * np.sqrt(omega[j]/omega_bath_m)
            g2_kj[j] = g_0 * f2_kj[j] * np.sqrt(omega[j]/omega_bath_p)

    f_kj = np.concatenate((f1_kj,f2_kj),axis=1)
    g_kj = np.concatenate((g1_kj,g2_kj),axis=1)
    om_B = np.concatenate((omega_bath_m,omega_bath_p))
    k_vals = np.concatenate((k_modes_m,k_modes_p))



    H_S[0:n_osc,n_osc:N] = -1j * g_kj
    H_S[n_osc:N,0:n_osc] = +1j * np.conjugate(np.transpose(g_kj))

    K[0:n_osc,n_osc:N] = 1j * g_kj
    K[n_osc:N,0:n_osc] = 1j * np.transpose(g_kj)

    alpha = np.zeros((n,n),dtype=np.complex_)
    beta = np.zeros((n,n),dtype=np.complex_)

    for k in range(n):
        for k_ in range(n):

            alpha[k][k_] = g_0**2 * np.transpose(np.conjugate(f_kj[:,k]))/np.sqrt(om_B[k]) @ (f_kj[:,k_]/np.sqrt(om_B[k_]))
            beta[k][k_] = g_0**2 * np.transpose(f_kj[:,k])/np.sqrt(om_B[k]) @ (f_kj[:,k_]/np.sqrt(om_B[k_]))


    Temp_zero_0 = np.zeros((N,N),dtype=np.complex_)
    Temp_zero_1 = np.zeros((n_osc,N),dtype=np.complex_)
    Temp_zero_2 = np.zeros((n,n_osc),dtype=np.complex_)


    α_tilde = np.concatenate((Temp_zero_1,np.concatenate((Temp_zero_2,alpha),axis=1)))
    β_tilde = np.concatenate((Temp_zero_1,np.concatenate((Temp_zero_2,beta),axis=1)))

    if RWA == 0:
        System_Bath = np.concatenate((np.concatenate((H_S,K)),np.concatenate((np.conjugate(K),np.conjugate(H_S)))),axis=1)

        Bath_Bath = np.concatenate((np.concatenate((α_tilde,β_tilde)),np.concatenate((np.conjugate(β_tilde),np.conjugate(α_tilde)))),axis=1)
    if RWA == 1:
        System_Bath = np.concatenate((np.concatenate((H_S,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(H_S)))),axis=1)

        Bath_Bath = np.concatenate((np.concatenate((α_tilde,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(α_tilde)))),axis=1)


    h = System_Bath + 2*Bath_Bath

    return [h, om_B, f_kj]

def Hamilton_Matrix_DG_WG(n,omega,omega_0,x,g_0,cd_choice,omega_c=0,spectrum='TB',RWA=0,strong=0,mode_offset=0,sm=1,HP=0.25, SS=1):
    n_osc = len(omega)
    sqrt_L = np.sqrt(omega_0*np.pi)
    L = omega_0*np.pi

    N = n_osc + n
    
    pi = np.pi
    
    if spectrum == 'linear':
        k_modes = 2*np.arange(mode_offset+1,n+mode_offset+1)/omega_0

        omega_bath = k_modes

    if spectrum == 'TB':
        k_modes_p = np.arange(0,(n*pi)/L,(2*pi)/L)
        k_modes_m = -np.flip(np.arange((2*pi)/L,(n*pi)/L+(2*pi)/L,(2*pi)/L))
        if omega_c == 0:
            omega_c = omega[0]
        omega_bath_p = omega_c - HP * np.cos(k_modes_p*(L/n))
        omega_bath_m = omega_c - HP * np.cos(k_modes_m*(L/n))


    f1_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)
    f2_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)
    g1_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)
    g2_kj = np.zeros((n_osc,int(n/2)),dtype=np.complex_)

    H_S = np.zeros((N,N),dtype=np.complex_)
    K = np.zeros((N,N),dtype=np.complex_)

    B_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    A_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    E_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    D_jl = np.zeros((n_osc,n_osc),dtype=np.complex_)
    omega_tot = np.concatenate((omega,omega_bath_m,omega_bath_p))
    
    H_S = H_S + np.diag(omega_tot)

    if cd_choice == 'dual_exp+-ikx':
        for i in range(n_osc):
            f1_kj[i] = np.exp(+1j * k_modes_m * x[i])
            f2_kj[i] = np.exp(+1j * k_modes_p * x[i])

    for j in range(n_osc):
        if strong == 0:
            g1_kj[j] = g_0 * f1_kj[j]
            g2_kj[j] = g_0 * f2_kj[j]

        if strong == 1:
            g1_kj[j] = g_0 * f1_kj[j] * np.sqrt(omega_bath_m/omega[j])
            g2_kj[j] = g_0 * f2_kj[j] * np.sqrt(omega_bath_p/omega[j])

    f_kj = np.concatenate((f1_kj,f2_kj),axis=1)
    g_kj = np.concatenate((g1_kj,g2_kj),axis=1)
    om_B = np.concatenate((omega_bath_m,omega_bath_p))
    k_vals = np.concatenate((k_modes_m,k_modes_p))


    H_S[0:n_osc,n_osc:N] = -1j * g_kj
    H_S[n_osc:N,0:n_osc] = +1j * np.conjugate(np.transpose(g_kj))


    K[0:n_osc,n_osc:N] = +1j * np.conjugate(g_kj)
    K[n_osc:N,0:n_osc] = +1j * np.conjugate(np.transpose(g_kj))

    for j in range(n_osc):
        for l in range(n_osc):
            B_jl[j][l] = np.imag((f_kj[j]/om_B) @ np.transpose(np.conjugate(f_kj[l])))

    B_squared = B_jl @ B_jl

    for j in range(n_osc):
        for l in range(n_osc):
            A_jl[j][l] = np.real(np.conjugate(f_kj[j]) @ np.transpose(f_kj[l])) - g_0**2 * B_squared[j][l]

    for j in range(n_osc):
        for l in range(n_osc):
            E_jl[j][l] = (1/np.sqrt(omega[j]*omega[l]))*(2*A_jl[j][l] + 1j * B_jl[j][l]*(omega[j]+omega[l]))

            D_jl[j][l] = (1/np.sqrt(omega[j]*omega[l]))*(2*A_jl[j][l] + 1j * B_jl[j][l]*(omega[j]-omega[l]))


    Temp_zero_0 = np.zeros((N,N),dtype=np.complex_)
    Temp_zero_1 = np.zeros((N,n),dtype=np.complex_)
    Temp_zero_2 = np.zeros((n,n_osc),dtype=np.complex_)

    E = np.concatenate((np.concatenate((E_jl,Temp_zero_2),axis=0),Temp_zero_1),axis=1)
    D = np.concatenate((np.concatenate((D_jl,Temp_zero_2),axis=0),Temp_zero_1),axis=1)

    if RWA == 0:
        System_Bath = np.concatenate((np.concatenate((H_S,np.conjugate(K))),np.concatenate((K,np.conjugate(H_S)))),axis=1)

        System_System = np.concatenate((np.concatenate((E,np.conjugate(D))),np.concatenate((D,np.conjugate(E)))),axis=1)
    if RWA == 1:
        System_Bath = np.concatenate((np.concatenate((H_S,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(H_S)))),axis=1)

        System_System = np.concatenate((np.concatenate((E,Temp_zero_0)),np.concatenate((Temp_zero_0,np.conjugate(E)))),axis=1)

    if SS == 1:
        h = System_Bath + g_0**2 * System_System
    else:
        h = System_Bath
        
    return [h, om_B, f_kj]

# Returns initial bath mode populations for a cavity in a thermal state
def n_bar(n,omega_0,beta,n_0=1):
    omR = np.zeros(n)
    for i in range(n):
        omR[i] = (i+1)/omega_0
    nbar = np.zeros(len(omR))
    if n_0==0:
        return nbar
    for i in range(0,len(omR)):
        nbar[i] = 1/(np.exp(beta*omR[i])-1)
    return nbar

def n_bar_TB(n,ωC,Hop,beta,n_0=1):
    L = n
    pi = np.pi
    omega_c = ωC
    k_modes_p = np.arange(0,(n*pi)/L,(2*pi)/L)
    k_modes_m = -np.flip(np.arange((2*pi)/L,(n*pi)/L+(2*pi)/L,(2*pi)/L))
    
    omega_bath_p = omega_c - Hop * np.cos(k_modes_p*(L/n))
    omega_bath_m = omega_c - Hop * np.cos(k_modes_m*(L/n))
    
    omR = np.concatenate((omega_bath_m,omega_bath_p))
    nbar = np.zeros(len(omR))
    if n_0==0:
        return nbar
    for i in range(0,len(omR)):
        nbar[i] = 1/(np.exp(beta*omR[i])-1)
    return nbar

# Returns ⟨𝜂†𝜂(0)⟩ for given initial bath correlations and given initial state of emitters
def Etad_Eta_matrix(n_osc,n_,initial,osc_phase = 0):
    d = len(n_osc) + len(n_)
    ηd_η = np.zeros((2*d,2*d),dtype=np.complex_)
    
    if initial == 'coherent':
        if osc_phase == 0:
            osc_phase = np.zeros(len(n_osc))
        for i in range(len(n_osc)):
            coh = n_osc[i]*cm.exp(complex(0,osc_phase[i]))
            ηd_η[i][i] = (abs(coh))**2
            ηd_η[i+d][i+d] = 1 + (abs(coh))**2
            ηd_η[i+d][i] = (coh)**2
            ηd_η[i][i+d] = (np.conj(coh))**2
            for j in range(len(n_osc)):
                coh2 = n_osc[j]*cm.exp(complex(0,osc_phase[j]))
                ηd_η[i][j] = np.conj(coh)*coh2
                ηd_η[i+d][j+d] = coh*np.conj(coh2)
                ηd_η[i+d][j] = coh*(coh2)
                ηd_η[i][j+d] = np.conj(coh)*np.conj(coh2)

    if initial == 'fock':
        for i in range(len(n_osc)):
            ηd_η[i][i] = n_osc[i]
            ηd_η[i+d][i+d] = 1 + n_osc[i]
            ηd_η[i+d][i] = 0
            ηd_η[i][i+d] = 0

    if initial == 'superposition+':
        for i in range(len(n_osc)):
            for j in range(len(n_osc)):
                ηd_η[i][j] = +0.5                     #To store initial values of a_i_dag_a_j
        for i in range(len(n_osc)):
            for j in range(len(n_osc)):
                if i == j:
                    ηd_η[i+d][j+d] = 1 + ηd_η[i][j] #To store initial values of a_i_a_j_dag
                else:
                    ηd_η[i+d][j+d] = ηd_η[j][i]
                ηd_η[i+d][j] = 0                      #To store initial values of a_i_a_j
                ηd_η[i][j+d] = 0                      #To store initial values of a_i_dag_a_j_dag

    if initial == 'superposition-':
        for i in range(len(n_osc)):
            for j in range(len(n_osc)):
                if i == j:
                    ηd_η[i][j] = +0.5                     #To store initial values of a_i_dag_a_j
                else:
                    ηd_η[i][j] = -0.5
        for i in range(len(n_osc)):
            for j in range(len(n_osc)):
                if i == j:
                    ηd_η[i+d][j+d] = 1 + ηd_η[i][j] #To store initial values of a_i_a_j_dag
                else:
                    ηd_η[i+d][j+d] = ηd_η[j][i]
                ηd_η[i+d][j] = 0                      #To store initial values of a_i_a_j
                ηd_η[i][j+d] = 0                      #To store initial values of a_i_dag_a_j_dag

    if initial == 'superposition1':
        for i in range(len(n_osc)):
            ηd_η[i][i] = 0.5
            ηd_η[i+d][i+d] = 1 + 0.5
            ηd_η[i+d][i] = 0
            ηd_η[i][i+d] = 0

    for i in range(len(n_osc),d):
        ηd_η[i][i] = n_[i-len(n_osc)]
        ηd_η[i+d][i+d] = 1 + n_[i-len(n_osc)]

    return ηd_η

def CavityHamiltonian(L,ωS,xS,N,g0,RWA_Switch=0,Broadband=0,gauge='d.E',mode_func='sinkx'):
    
    Δω = np.pi/L
    
    if Broadband == 1:
        strc = 0
    else:
        strc = 1

    if gauge == 'd.E':
        return Hamilton_Matrix_dG(N,ωS,1/Δω,xS,g0,mode_func,RWA=RWA_Switch,strong=Broadband)
    if gauge == 'coulomb':
        g0 = g0 * np.ones(len(xS),dtype=np.complex_)
        return Hamilton_Matrix_CG(N,ωS,1/Δω,xS,g0,mode_func,RWA=RWA_Switch,strong=Broadband)[0]
    if gauge == 'dipole':
        return Hamilton_Matrix_DG(N,ωS,1/Δω,xS,g0,mode_func,RWA=RWA_Switch,strong=Broadband)[0]

def CCAHamiltonian(ωS,xS,ωC,Hop,N,g0,RWA_Switch=0,Broadband=0,gauge='d.E'):
    if Broadband == 1:
        strc = 0
    else:
        strc = 1
    
    if gauge == 'coulomb':
        return Hamilton_Matrix_CG_WG(N,ωS,N/np.pi,xS,g0,
                                     'dual_exp+-ikx',omega_c=ωC,spectrum='TB',
                                     RWA=RWA_Switch,strong=strc,mode_offset=0,HP=Hop)[0]
    if gauge == 'dipole':
        return Hamilton_Matrix_DG_WG(N,ωS,N/np.pi,xS,g0,
                                     'dual_exp+-ikx',omega_c=ωC,spectrum='TB',
                                     RWA=RWA_Switch,strong=strc,mode_offset=0,HP=Hop)[0]

    if gauge == 'd.E':
        return Hamilton_Matrix_DG_WG(N,ωS,N/np.pi,xS,g0,
                                     'dual_exp+-ikx',omega_c=ωC,spectrum='TB',
                                     RWA=RWA_Switch,strong=strc,mode_offset=0,HP=Hop,SS=0)[0]

def CavityInitCorr(N,L,cavity_state,dipole_state,N_d,phase=0,Beta=200,pop=1):
    
    if cavity_state == 'thermal':
        N_ = n_bar(N,np.pi/L,Beta,n_0=1)
    if cavity_state == 'vacuum':
        N_ = n_bar(N,np.pi/L,Beta,n_0=0)
    n_d = pop * np.ones(N_d)
    if phase == None:
        phase = np.zeros(N_d)
    
    X = Etad_Eta_matrix(n_d,N_,dipole_state,osc_phase = phase)

    return X

def CCAInitCorr(N,ωC,Hop,cca_state,dipole_state,N_d,phase=None,Beta=200,pop=1):
    
    if cca_state == 'thermal':
        N_ = n_bar_TB(N,ωC,Hop,Beta,n_0=1)
    if cca_state == 'vacuum':
        N_ = n_bar_TB(N,ωC,Hop,Beta,n_0=0)

    if phase == None:
        phase = np.zeros(N_d)
    
    n_d = pop * np.ones(N_d)
    
    X = Etad_Eta_matrix(n_d,N_,dipole_state,osc_phase = phase)
    
    return X

def CavityInitLace(Init,N,L,cavity_state,Beta=200):
    if cavity_state == 'thermal':
        N_ = n_bar(N,np.pi/L,Beta,n_0=1)
    if cavity_state == 'vacuum':
        N_ = n_bar(N,np.pi/L,Beta,n_0=0)
    N_d = int(len(Init)/2)
    d = N_d + N
    Etd_Et = np.zeros((2*d,2*d),dtype=np.complex_)
    
    for i in range(N_d):
        for j in range(N_d):
            Etd_Et[i][j] = Init[i][j]
            Etd_Et[i][j+d] = Init[i][j+N_d]
            Etd_Et[i+d][j] = Init[i+N_d][j]
            Etd_Et[i+d][j+d] = Init[i+N_d][j+N_d]
    
    for i in range(N_d,d):
        Etd_Et[i][i] = N_[i-N_d]
        Etd_Et[i+d][i+d] = 1 + N_[i-N_d]

    return Etd_Et

def CCAInitLace(Init,N,ωC,Hop,cca_state,Beta=200):
    if cca_state == 'thermal':
        N_ = n_bar_TB(N,ωC,Hop,Beta,n_0=1)
    if cca_state == 'vacuum':
        N_ = n_bar_TB(N,ωC,Hop,Beta,n_0=0)
    N_d = int(len(Init)/2)
    d = N_d + len(N_)
    Etd_Et = np.zeros((2*d,2*d),dtype=np.complex_)
    
    for i in range(N_d):
        for j in range(N_d):
            Etd_Et[i][j] = Init[i][j]
            Etd_Et[i][j+d] = Init[i][j+N_d]
            Etd_Et[i+d][j] = Init[i+N_d][j]
            Etd_Et[i+d][j+d] = Init[i+N_d][j+N_d]
    
    for i in range(N_d,d):
        Etd_Et[i][i] = N_[i-N_d]
        Etd_Et[i+d][i+d] = 1 + N_[i-N_d]
    
    return Etd_Et

def ArrayEvolve(H,Adag_A0,N_d,t_end,dt=0.01):
    T, U, λ = Bogoliubov_Transform(H)
    
    T = np.array(T,dtype=np.complex_)
    
    A = np.linalg.inv(T)
    
    zeta_d_zeta = Zetad_Zeta_Matrix(T,Adag_A0)
    All_Corr = []
    for i in range(N_d):
        modified_Zeta_Corr = mod_zeta_corr(A,zeta_d_zeta,i,i)
        
        initial_val_error = np.abs(Adag_A0[i][i]-Eta_correlation_t(λ,modified_Zeta_Corr,0))
        
        if initial_val_error > 1.e-13:
            print('Initial Value error: ',initial_val_error)
        
        Time, Corr = Evolve(λ,modified_Zeta_Corr,t_end,del_t = dt)
        
        if i == 0:
            Meta_Corr = np.zeros(len(Time),dtype=np.complex_) 
        
        Meta_Corr += Corr
        All_Corr.append(Corr)
        
    return [Time, Meta_Corr, All_Corr]

def Bath_correlators(m,n,res_mode,diag,invBog,zd_z_m,t,Nb=1,Mul_mode=10):
    
    zeta_d_zeta = zd_z_m
    
    Bath_corr = np.zeros((2*n,2*n),dtype = np.complex_)
    for i in range(res_mode-Mul_mode,res_mode+Mul_mode):
        for j in range(res_mode-Mul_mode,res_mode+Mul_mode):
            p = m + i
            q = m + j
            mod_ZZ_corr = mod_zeta_corr(invBog,zeta_d_zeta,p-1,q-1)

            Bath_corr[i][j] = Eta_correlation_t(diag,mod_ZZ_corr,t)

            p = 2*m + n + i
            q = m + j
            mod_ZZ_corr = mod_zeta_corr(invBog,zeta_d_zeta,p-1,q-1)

            Bath_corr[i+n][j] = Eta_correlation_t(diag,mod_ZZ_corr,t)

            Bath_corr[i][j+n] = np.conjugate(Bath_corr[i+n][j])

            Bath_corr[j+n][i+n] = Bath_corr[i][j] # + delta(i,j)
#             print(i,j)
    mat2 = np.ones((n,n))
    mat1 = np.concatenate((np.concatenate((-mat2,mat2)), np.concatenate((mat2,-mat2))),axis=1)
    BC = Bath_corr * mat1

    return BC

def intensity_sin(omega_0,R_corr):
    n = int(len(R_corr)/2)
    omR = np.zeros(n)
    for i in range(n):
        omR[i] = (i+1)/omega_0
    gk = np.zeros(n,dtype = np.complex_)
    x = np.arange(0,omega_0*np.pi,0.05)
    I_ = np.zeros(len(x), dtype = np.complex_)
    
    for i in range(len(x)):
        gk = np.sqrt(omR) * np.sin(x[i]*omR)
        gk_conj = gk
        gk_dag = np.concatenate((gk_conj,gk))
        gk = np.transpose(np.conjugate(gk_dag))
        I_[i] = gk_dag @ R_corr @ gk
        
    I = I_/(omega_0*np.pi)

    return (x,I)

def CavityIntensity(ω,Nb,L,h,Ad_A0,t_vals,multi_mode=15):
    
    T, U, λ = Bogoliubov_Transform(h)

    ωS = np.average(ω)
    ω0 = L/np.pi
    res_mode = int(ωS * ω0)
    
    T = np.array(T,dtype=np.complex_)
    
    A = np.linalg.inv(T)
    
    zeta_d_zeta = Zetad_Zeta_Matrix(T,Ad_A0)

    I_data = []

    for i in range(len(t_vals)):
        print('Calculating Electric field Intensity at t = ',t_vals[i], end="\r")
        B = Bath_correlators(len(ω),Nb,res_mode,λ,A,
                             zeta_d_zeta,t_vals[i],Mul_mode=multi_mode)
        X_axis,Int = intensity_sin(ω0, B)

        I_data.append(Int)

    return [X_axis,I_data]

def Position_Space(B,k_vals,x_vals):
    m = len(k_vals)
    Pop_x = np.zeros(m,dtype=np.complex_)
    for i in range(m):
        x = x_vals[i]
        Λ = np.exp( 1j * k_vals * x)
        Pop_x[i] = np.conjugate(np.transpose(Λ)) @ B @ Λ

    return Pop_x

def CCAPopulation(ω,N,ωC,Hop,h,Ad_A0,t_vals,leave_modes=10):
    T, U, λ = Bogoliubov_Transform(h)

    T = np.array(T,dtype=np.complex_)
    
    A = np.linalg.inv(T)
    
    zeta_d_zeta = Zetad_Zeta_Matrix(T,Ad_A0)

    pi, a, L = np.pi, 1, N
    k_modes_p = np.arange(0,(N*pi)/L,(2*pi)/L)
    k_modes_m = -np.flip(np.arange((2*pi)/L,(N*pi)/L+(2*pi)/L,(2*pi)/L))
    k = np.concatenate((k_modes_m,k_modes_p))
    x = np.arange(-int(N/2),int(N/2),1)*a
    m = len(ω)

    Pop_X_t = []
    
    for t in t_vals:
        print('Calculating photonic population distribution at t = ',t, end="\r")
        Bath_corr = np.zeros((1*N,1*N),dtype = np.complex_)
        for i in range(leave_modes,N-leave_modes):
            for j in range(leave_modes,N-leave_modes):
                p = m + i
                q = m + j
                mod_ZZ_corr = mod_zeta_corr(A,zeta_d_zeta,p-1,q-1)
    
                Bath_corr[i][j] = Eta_correlation_t(λ,mod_ZZ_corr,t)

        Pop_X = Position_Space(Bath_corr, k, x)
        Pop_X_t.append(Pop_X)
    return [x,Pop_X_t]