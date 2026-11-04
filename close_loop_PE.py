import pandas as pd
import os
import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.integrate import ode
from math import exp, log, pi
import time
import math
import joblib
from scipy import stats
from pyDOE2 import lhs, fullfact
import sympy as sp
from numpy.polynomial.legendre import Legendre
import tempfile
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn import preprocessing
from sklearn.utils import shuffle
from numpy.random import seed
from sklearn.metrics import mean_squared_error, r2_score
import glob
from scipy.stats import qmc
from scipy.optimize import minimize
import csv
from argparse import ArgumentParser, FileType
from io import StringIO

def legendre_gauss_lobatto(N):
    """
    Compute Legendre-Gauss-Lobatto (LGL) nodes.
    N = polynomial degree
    Returns N+1 nodes in [-1,1]
    """
    if N == 1:
        return np.array([-1.0, 1.0])
    # Legendre polynomial of degree N
    Pn = Legendre.basis(N)
    # Derivative
    dPn = Pn.deriv()
    # Roots of derivative give interior LGL points
    interior = dPn.roots()
    # Include endpoints
    x = np.hstack((-1.0, interior, 1.0))
    return np.sort(x)


def differentiation_matrix(x):
    """
    Construct first differentiation matrix using barycentric formula.
    x: collocation points
    """
    N = len(x) - 1
    D = np.zeros((N + 1, N + 1))

    # Compute barycentric weights
    w = np.ones(N + 1)
    for j in range(N + 1):
        for k in range(N + 1):
            if k != j:
                w[j] /= (x[j] - x[k])

    # Fill matrix
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = w[j] / (w[i] * (x[i] - x[j]))

    # Diagonal entries
    for i in range(N + 1):
        D[i, i] = -np.sum(D[i, :])

    return D


def scaled_matrices(N, a=-1, b=1):
    """
    Returns:
    x  = LGL nodes in [a,b]
    D  = first derivative matrix on [a,b]
    D2 = second derivative matrix on [a,b]
    """
    # LGL nodes in [-1,1]
    x_ref = legendre_gauss_lobatto(N)

    # Map to [a,b]
    x = 0.5 * (b - a) * x_ref + 0.5 * (a + b)

    # Differentiation matrix in reference domain
    D_ref = differentiation_matrix(x_ref)

    # Scale to [a,b]
    D = 2 / (b - a) * D_ref
    D2 = D @ D

    return x, D, D2


def est_U(Q_main,T_j,TVR, a_d):

    Flowrate_w = 3  # L/min
    Flowrate_w_si = Flowrate_w * 1e-3 / 60  # m^3/s

    # Thermal properties of water
    # https://www.myengineeringtools.com/Data_Diagrams/Water_Properties_Correlations.html
    a_w = -2.8054253e-10
    b_w = 1.0556302e-7
    c_w = -4.6170461e-5
    d_w = -0.0079870401
    e_w = 16.945176
    f_w = 999.83952
    g_w = 0.01687985
    T_c = 20

    A_mu_w = -3.7188
    B_mu_w = 578.919
    C_mu_w = -137.546

    cp_1_w = 276370
    cp_2_w = -2090.1
    cp_3_w = 8.125
    cp_4_w = -0.014116
    cp_5_w = 9.3701e-6

    cp_1_m = 256040
    cp_2_m = -2741.4
    cp_3_m = 14.777
    cp_4_m = -0.035078
    cp_5_m = 3.27191e-5

    # Jacket cooling water stream properties at 1 bar
    # mu_w = 1.1375e-3 # units: Ns/m^2
    cp_w = 4200  # J/(kg.K)
    rho_w = 1000  # kG/m^3
    k_w = 666.84e-3  # W/(K.m^2)

    # reactor methanol stream properties at 1 bar
    mu_m = 0.56e-3  # units: Ns/m^2
    cp_m = 2534  # J/(kg.K)
    rho_m = 792  # kG/m^3
    k_m = 0.204  # W/(K.m^2)

    # reactor geometry
    L = 104e-3  # length of reactor in m
    d_rotor = 19.0e-3  # diameter of rotor
    d_in = 20.4e-3  # internal diameter of glass
    d_out = 24.4e-3  # internal diameter of glass
    d_j = 28.4e-3  # assumed jacket cross-section diameter

    t_j = 1.86  # heat transfer factor
    t_r = 1.86  # heat transfer factor
    vis_cor_r = mu_m / mu_m

    # solid conductive heat transfer
    #k_g = 1.05  # W/(K.m^2)
    k_g = 15  # W/(K.m^2) steel
    R_s = d_out * math.log(d_out / d_in) / (2 * k_g)

    # jacket inside fluid dynamics
    Aj = math.pi * (d_j ** 2 - d_out ** 2) / 4  # jacket cross-section area
    Dje = d_j - d_out  # jacket inside effective diameter

    # reactor inside fluid dynamics
    Ar = math.pi * (d_in ** 2 - d_rotor ** 2) / 4  # reactor cross-section area
    Dre = d_in - d_rotor  # reactor inside effective diameter

    T_c = T_j # degree celsius
    rho_w = (((((a_w * T_c + b_w) * T_c + c_w) * T_c + d_w) * T_c + e_w) * T_c + f_w) / (1 + g_w * T_c)
    mass_fl_w = Flowrate_w_si * rho_w

    rotor_rev  = TVR/60
    rotor_w = rotor_rev*2*math.pi
    rotor_v = rotor_w * d_rotor / 2

    Flowrate_m = Q_main  # uL/s
    Flowrate_m_si = Flowrate_m * 1e-9  # m^3/s
    mass_fl_m = Flowrate_m_si * rho_m

    cp_w = (cp_1_w + cp_2_w * (T_c + 273.15) + cp_3_w * (T_c + 273.15) ** 2 + cp_4_w * (
                T_c + 273.15) ** 3 + cp_5_w * (T_c + 273.15) ** 4) / 18
    cp_m = (cp_1_m + cp_2_m * (T_c + 273.15) + cp_3_m * (T_c + 273.15) ** 2 + cp_4_m * (
                T_c + 273.15) ** 3 + cp_5_m * (T_c + 273.15) ** 4) / 32

    mu_w = 1e-3 * math.exp(A_mu_w + B_mu_w / (C_mu_w + (T_c + 273.15)))

    vis_cor_j = mu_w / mu_w

    # jacket inside fluid dynamics
    Gj = mass_fl_w / Aj  # jacket fluid mass velocity
    #Re_j = Gj * Dje / mu_w  # jacket Reynold's number
    Re_j = rho_w * rotor_w * d_rotor / 2 * Dje / mu_w  # annulus Reynold's number
    Pr_j = cp_w * mu_w / k_w  # Prandtl's number

    # reactor inside fluid dynamics
    Gr = mass_fl_m / Ar  # reactor fluid mass velocity
    Re_r = rho_m * rotor_w * d_rotor/2 * Dre / mu_m  # annulus Reynold's number
    Pr_r = cp_m * mu_m / k_m  # Prandtl's number

    h_j = heat_transfer(Re_j, Pr_j, vis_cor_j, k_w, Dje, L)
    h_r = heat_transfer(Re_r, Pr_r, vis_cor_r, k_m, Dre, L)
    U_inv = 1 / h_j + R_s + d_out / (d_in * h_r)
    U_re_i = Re_r
    U_re_o = Re_j
    U_h_i = h_r*d_in/d_out
    U_h_o = h_j
    U = 1 / U_inv
    T_j = (T_j-20)/11.33
    Ta = (TVR*2*pi/60)*d_rotor/2*(Dre/2) / (mu_m/rho_m)
    Re = Q_main/Ar*rho_m*Dre/mu_m
    x_1 = (Ta**2 -138710)/141571 #scalers
    x_2 = (Ta*Re -1251)/1067 #scalers
    x_3 = (Re**2 -20.95)/16.9 #scalers
    dU = (a_d[0]*x_1 + a_d[1]*x_2)*U #+ a_d[2]*x_3

    U_act = U - dU

    if U_act < 0.1:
        U_act = 0.1
    elif U_act > 1000:
        U_act = 1000
    return U_act

def heat_transfer_coeff_turbulent(Re,Pr, vis_cor,k,De):
    h = 0.023*Re**0.8*(Pr)**0.33*(vis_cor)**(0.14)*(k/De)
    return h

def heat_transfer_coeff_laminar(Re,Pr, vis_cor,k,De,L):
    h = 1.86*(Re*Pr)**0.33*(De/L)**0.33*(vis_cor)**(0.14)*(k/De)
    return h

def heat_transfer(Re,Pr, vis_cor,k,De,L):
    if Re <=2000.0:
        h = heat_transfer_coeff_laminar(Re,Pr, vis_cor,k,De,L)
    else:
        h =  heat_transfer_coeff_turbulent(Re,Pr, vis_cor,k,De)
    return h

def adm_nonadiabatic(x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[0]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58 *h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    #U=1
    #U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)*1

    xl, D_m, D2_m = scaled_matrices(N_step, 0, l_r)
    oc_ca =[(Da*D2_m[1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) - \
         kref*exp(Ea/R*(1/(T[0]+273)-1/(Tref1+273)))*ca[0]*(cb0-(ca0-ca[0]))).item()]
    oc_ca_others = [(Da*D2_m[i+1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[i+1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) - \
         kref*exp(Ea/R*(1/(T[i]+273)-1/(Tref1+273)))*ca[i]*(cb0-(ca0-ca[i]))).item()       for i in range(1,N_step-1)
    ]
    oc_ca_1 = [(D_m[N_step,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()]
    oc_ca_others.extend(oc_ca_1)
    oc_ca.extend(oc_ca_others)
    oc_T = [(-lambda_a*Ar*D2_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[0])*cpa+(cb0-(ca0-ca[0]))*cpb+(ca0-ca[0])*cpc+(ca0-ca[0])*cpd+mol_m*cpm)\
          *D_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[0])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i))-Ar*(DHr+Dcp*(T[0]-Tref2))*
                      kref*exp(Ea/R*(1/(T[0]+273)-1/(Tref1+273)))*ca[0]*(cb0-(ca0-ca[0])))).item()]
    oc_T_others =[(-lambda_a*Ar*D2_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[i])*cpa+(cb0-(ca0-ca[i]))*cpb+(ca0-ca[i])*cpc+(ca0-ca[i])*cpd+mol_m*cpm)\
          *D_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[i])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i))-Ar*(DHr+Dcp*(T[i]-Tref2))*
                      kref*exp(Ea/R*(1/(T[i]+273)-1/(Tref1+273)))*ca[i]*(cb0-(ca0-ca[i])))).item()  for i in range(1,N_step-1)]
    oc_T_1 = [(D_m[N_step,:]@ np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))).item()]

    oc_T_others.extend(oc_T_1)
    oc_T.extend(oc_T_others)
    oc_ca.extend(oc_T)
    return oc_ca

def adm_noreaction(x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[0]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58 *h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    #U=1
    #U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n) *1

    xl, D_m, D2_m = scaled_matrices(N_step, 0, l_r)
    oc_ca =[(Da*D2_m[1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()]
    oc_ca_others = [(Da*D2_m[i+1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[i+1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()       for i in range(1,N_step-1)
    ]
    oc_ca_1 = [(D_m[N_step,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()]
    oc_ca_others.extend(oc_ca_1)
    oc_ca.extend(oc_ca_others)
    oc_T = [(-lambda_a*Ar*D2_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[0])*cpa+(cb0-(ca0-ca[0]))*cpb+(ca0-ca[0])*cpc+(ca0-ca[0])*cpd+mol_m*cpm)\
          *D_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[0])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i)))).item()]
    oc_T_others =[(-lambda_a*Ar*D2_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[i])*cpa+(cb0-(ca0-ca[i]))*cpb+(ca0-ca[i])*cpc+(ca0-ca[i])*cpd+mol_m*cpm)\
          *D_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[i])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i)))).item()  for i in range(1,N_step-1)]
    oc_T_1 = [(D_m[N_step,:]@ np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))).item()]

    oc_T_others.extend(oc_T_1)
    oc_T.extend(oc_T_others)
    oc_ca.extend(oc_T)
    return oc_ca

def adm_noreaction_atm(x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[5]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58 *h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    #U=1
    #U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    xl, D_m, D2_m = scaled_matrices(N_step, 0, l_r)
    oc_ca =[(Da*D2_m[1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()]
    oc_ca_others = [(Da*D2_m[i+1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[i+1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()       for i in range(1,N_step-1)
    ]
    oc_ca_1 = [(D_m[N_step,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()]
    oc_ca_others.extend(oc_ca_1)
    oc_ca.extend(oc_ca_others)
    oc_T = [(-lambda_a*Ar*D2_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[0])*cpa+(cb0-(ca0-ca[0]))*cpb+(ca0-ca[0])*cpc+(ca0-ca[0])*cpd+mol_m*cpm)\
          *D_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[0])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i)))).item()]
    oc_T_others =[(-lambda_a*Ar*D2_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[i])*cpa+(cb0-(ca0-ca[i]))*cpb+(ca0-ca[i])*cpc+(ca0-ca[i])*cpd+mol_m*cpm)\
          *D_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[i])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i)))).item()  for i in range(1,N_step-1)]
    oc_T_1 = [(D_m[N_step,:]@ np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))).item()]

    oc_T_others.extend(oc_T_1)
    oc_T.extend(oc_T_others)
    oc_ca.extend(oc_T)
    return oc_ca

def adm_adiabatic(x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[0]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58 *h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    #U=1
    #U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)
    U=0

    xl, D_m, D2_m = scaled_matrices(N_step, 0, l_r)
    oc_ca =[(Da*D2_m[1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) - \
         kref*exp(Ea/R*(1/(T[0]+273)-1/(Tref1+273)))*ca[0]*(cb0-(ca0-ca[0]))).item()]
    oc_ca_others = [(Da*D2_m[i+1,:]@np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) \
    - u*D_m[i+1,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1))) - \
         kref*exp(Ea/R*(1/(T[i]+273)-1/(Tref1+273)))*ca[i]*(cb0-(ca0-ca[i]))).item()       for i in range(1,N_step-1)
    ]
    oc_ca_1 = [(D_m[N_step,:]@ np.concatenate((np.array(ca0).reshape(1,1),ca.reshape(-1,1)))).item()]
    oc_ca_others.extend(oc_ca_1)
    oc_ca.extend(oc_ca_others)
    oc_T = [(-lambda_a*Ar*D2_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[0])*cpa+(cb0-(ca0-ca[0]))*cpb+(ca0-ca[0])*cpc+(ca0-ca[0])*cpd+mol_m*cpm)\
          *D_m[1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[0])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i))-Ar*(DHr+Dcp*(T[0]-Tref2))*
                      kref*exp(Ea/R*(1/(T[0]+273)-1/(Tref1+273)))*ca[0]*(cb0-(ca0-ca[0])))).item()]
    oc_T_others =[(-lambda_a*Ar*D2_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))+\
          F_total*((ca[i])*cpa+(cb0-(ca0-ca[i]))*cpb+(ca0-ca[i])*cpc+(ca0-ca[i])*cpd+mol_m*cpm)\
          *D_m[i+1,:]@np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1))) - \
          (U*pi*2*r_o*(Tc-T[i])+1/8*(pi*mu_m*w**2*(2*r_o)**3)/(2*(r_o-r_i))-Ar*(DHr+Dcp*(T[i]-Tref2))*
                      kref*exp(Ea/R*(1/(T[i]+273)-1/(Tref1+273)))*ca[i]*(cb0-(ca0-ca[i])))).item()  for i in range(1,N_step-1)]
    oc_T_1 = [(D_m[N_step,:]@ np.concatenate((np.array(T0).reshape(1,1),T.reshape(-1,1)))).item()]

    oc_T_others.extend(oc_T_1)
    oc_T.extend(oc_T_others)
    oc_ca.extend(oc_T)
    return oc_ca

def PFR_nonadiabatic(t,x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[0]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58*h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    U=1
    U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    return [-kref/u*exp(Ea/R*(1/(T+273)-1/(Tref1+273)))*ca*(cb0-(ca0-ca)),
            (1/(F_total*((ca)*cpa+(cb0-(ca0-ca))*cpb+(ca0-ca)*cpc+(ca0-ca)*cpd+mol_m*cpm)))*(U * pi * 2 * r_o * (Tc -T) + 1 / 8 * (pi * mu_m * w ** 2 * (2 * r_o) ** 3) / (2 * (r_o - r_i)) - Ar * (
                        DHr + Dcp * (T - Tref2)) *
             kref * exp(Ea / R * (1 / (T + 273) - 1 / (Tref1 + 273))) * ca * (cb0 - (ca0 - ca)))
            ]

def PFR_noreaction_atm(t,x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[5]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58*h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    U=1
    U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    return [ca-ca,
            (1/(F_total*(ca*cpa+(cb0-(ca0-ca))*cpb+(ca0-ca)*cpc+(ca0-ca)*cpd+mol_m*cpm)))*(U * pi * 2 * r_o * (Tc -T) + 1 / 8 * (pi * mu_m * w ** 2 * (2 * r_o) ** 3) / (2 * (r_o - r_i)))
            ]

def PFR_noreaction(t,x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[0]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58*h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    U=1
    U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    return [ca-ca,
            (1/(F_total*(ca*cpa+(cb0-(ca0-ca))*cpb+(ca0-ca)*cpc+(ca0-ca)*cpd+mol_m*cpm)))*(U * pi * 2 * r_o * (Tc -T) + 1 / 8 * (pi * mu_m * w ** 2 * (2 * r_o) ** 3) / (2 * (r_o - r_i)))
            ]

def PFR_adiabatic(t,x_out,x_in,u_meas,theta,h_scale,l_r):
    N_step = int(len(x_out)/2)
    ca=x_out[:N_step]
    T = x_out[N_step:]
    #L = 0.231 #m
    h = l_r/N_step
    u=u_meas[2]
    Da = u_meas[3]
    ca0=x_in[0] #mol/m3
    cb0 = x_in[1] #mol/m3
    Tc = u_meas[0]
    T0 = x_in[4]
    tau = u_meas[1] #s
    w=u_meas[4]*2*pi/60
    R=8.31
    Tref1 = 25
    Tref2 = 25
    kref = theta[0]
    Ea = theta[1]

    r_i = 9.5e-3
    r_o = 10.2e-3

    Ar = pi*(r_o**2-r_i**2)
    lambda_a = theta[2]
    mwa = 106.12 #g/mol
    mwb = 107.15 #g/mol
    mwd = 18 #g/mol
    mwc = mwa + mwb - mwd
    cpa = 172    #J/(mol.K)
    cpb = 207.18       #J/(mol.K)
    cpc = 1.38 * mwc  # J/(mol.K)
    cpd = 4.184 * mwd  # J/(mol.K)
    mu_m = 0.56e-3  # units: Ns/m^2
    cpm = 2534  # J/(kg.K)
    mw_m = 0.032  # kg/mol
    rho_m = 792  # kg/m^3
    mol_m = rho_m/mw_m #mol/m^3

    har = -87.1*h_scale      #J/mol
    hbr = 34.2*h_scale     #J/mol
    hcr = 183.78 *h_scale    #J/mol
    hdr = -285.58*h_scale   #J/(mol.K)
    DHr =  hdr + hcr - hbr - har
    Dcp = cpd + cpc - (cpb + cpa)

    ((ca0-ca[0])*cpa+(cb0-(cb0-ca[0]))*cpb+ca[0]*cpc+ca[0]*cpd)
    F_total = 21*1e-6/tau #m3/s
    T_j = u_meas[0]
    TVR = u_meas[4]
    a_d = theta[3:]
    U=1
    U = est_U(F_total, T_j, TVR, a_d)
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)
    U=0

    return [-kref/u*exp(Ea/R*(1/(T+273)-1/(Tref1+273)))*ca*(cb0-(ca0-ca)),
            (1/(F_total*((ca)*cpa+(cb0-(ca0-ca))*cpb+(ca0-ca)*cpc+(ca0-ca)*cpd+mol_m*cpm)))*(U * pi * 2 * r_o * (Tc-T) + 1 / 8 * (pi * mu_m * w ** 2 * (2 * r_o) ** 3) / (2 * (r_o - r_i)) - Ar * (
                        DHr + Dcp * (T - Tref2)) *
             kref * exp(Ea / R * (1 / (T + 273) - 1 / (Tref1 + 273))) * ca * (cb0 - (ca0 - ca)))
            ]

def para_gen(pm, ps, p_samples,method):
    #pm = [3.47, 2.75]
    #ps = [0.05, 0.28]
    if method ==0:
        parameter_bounds = [pm[0]-ps[0], pm[0]+ps[0], pm[1]-ps[1], pm[1]+ps[1], pm[2]-ps[2], pm[2]+ps[2], pm[3]-ps[3], pm[3]+ps[3], pm[4]-ps[4], pm[4]+ps[4]]  # Bounds on aromatic substitution parameter set
        p_b = np.array(parameter_bounds).reshape(5, 2)
        #p_samples = 200
        n_phi = p_b.shape[0]  #

        # This code generates DoE by Latin hypercube sampling
        prng = 3
        lhs_points = lhs(n_phi, samples=p_samples, criterion='maximin', iterations=50, random_state=prng)
        paramset = np.zeros([p_samples, n_phi])
        for i in range(p_samples):
            paramset[i, 0] = lhs_points[i, 0] * (p_b[0, 1] - p_b[0, 0]) + p_b[0, 0]
            paramset[i, 1] = lhs_points[i, 1] * (p_b[1, 1] - p_b[1, 0]) + p_b[1, 0]
            paramset[i, 2] = lhs_points[i, 2] * (p_b[2, 1] - p_b[2, 0]) + p_b[2, 0]
            paramset[i, 3] = lhs_points[i, 3] * (p_b[3, 1] - p_b[3, 0]) + p_b[3, 0]
            paramset[i, 4] = lhs_points[i, 4] * (p_b[4, 1] - p_b[4, 0]) + p_b[4, 0]

    elif method ==1:
        dependency = np.diag((ps))
        latent = np.random.randn(p_samples, len(pm))
        dependent = latent.dot(dependency)
        paramset = dependent + pm
        #paramset[paramset < 0] = 0
    return paramset

def U_2(Flowrate, TVR, c,m,n):
    # Fluid parameters
    rho_m = 792  # kg/m^3
    mu_m = 0.56e-3  # units: Ns/m^2
    k_m = 0.204  # W/(K.m^2)

    r_i = 9.5e-3
    r_o = 10.2e-3

    # reactor geometry
    Dh =2*(r_o-r_i)  # reactor inside effective diameter

    # reactor inside fluid dynamics
    Ar = math.pi * (r_o ** 2 - r_i ** 2)  # reactor cross-section area
    omega = 2*math.pi*TVR/60
    vel = Flowrate/Ar
    Ta = omega*r_i*(r_o-r_i)/(mu_m/rho_m)

    Re = vel*(r_o-r_i)/(mu_m/rho_m)
    Nu = c *Re**m*Ta**n
    U=k_m/Dh*Nu

    if U< 1:
        U = 1
    elif U>100:
        U = 100
    return U

def adm_pfr_na(x_in, u_meas, theta, h_scale, l,n_steps):
    dl = l / (int(n_steps / 2))
    x = [x_in[0], x_in[4]]
    r = ode(PFR_nonadiabatic).set_integrator('vode', method='bdf')
    r.set_initial_value(x, 0).set_f_params(x_in, u_meas, theta, h_scale, l)
    l_n = np.linspace(0, l, int(n_steps / 2))
    ca_p = []
    T_p = []
    # for i in range(int(n/2)):
    while r.successful() and r.t < l:
        xf = r.integrate(r.t + dl)
        ca_p.append(xf[0])
        T_p.append(xf[1])

    cat = fsolve(adm_nonadiabatic, [ca_p[1:], T_p[1:]], args=(x_in, u_meas, theta, h_scale, l))
    ca_adm = cat[:int(len(cat) / 2)] / 1000
    T_adm = cat[int(len(cat) / 2):]

    tau = u_meas[1] #s
    TVR = u_meas[4]
    F_total = 21*1e-6/tau #m3/s
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)
    return ca_adm,T_adm,U

def adm_pfr_nr_atm(x_in, u_meas, theta, h_scale, l,n_steps):
    dl = l / (int(n_steps / 2))
    x = [x_in[0], x_in[4]]
    r = ode(PFR_noreaction_atm).set_integrator('vode', method='bdf')
    r.set_initial_value(x, 0).set_f_params(x_in, u_meas, theta, h_scale, l)
    l_n = np.linspace(0, l, int(n_steps / 2))
    ca_p = []
    T_p = []
    # for i in range(int(n/2)):
    while r.successful() and r.t < l:
        xf = r.integrate(r.t + dl)
        ca_p.append(xf[0])
        T_p.append(xf[1])

    cat = fsolve(adm_noreaction_atm, [ca_p[1:], T_p[1:]], args=(x_in, u_meas, theta, h_scale, l))
    ca_adm = cat[:int(len(cat) / 2)] / 1000
    T_adm = cat[int(len(cat) / 2):]

    tau = u_meas[1] #s
    TVR = u_meas[4]
    F_total = 21*1e-6/tau #m3/s
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    return ca_adm,T_adm,U

def adm_pfr_nr(x_in, u_meas, theta, h_scale, l,n_steps):
    dl = l / (int(n_steps / 2))
    x = [x_in[0], x_in[4]]
    r = ode(PFR_noreaction).set_integrator('vode', method='bdf')
    r.set_initial_value(x, 0).set_f_params(x_in, u_meas, theta, h_scale, l)
    l_n = np.linspace(0, l, int(n_steps / 2))
    ca_p = []
    T_p = []
    # for i in range(int(n/2)):
    while r.successful() and r.t < l:
        xf = r.integrate(r.t + dl)
        ca_p.append(xf[0])
        T_p.append(xf[1])

    cat = fsolve(adm_noreaction, [ca_p[1:], T_p[1:]], args=(x_in, u_meas, theta, h_scale, l))
    ca_adm = cat[:int(len(cat) / 2)] / 1000
    T_adm = cat[int(len(cat) / 2):]

    tau = u_meas[1] #s
    TVR = u_meas[4]
    F_total = 21*1e-6/tau #m3/s
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    return ca_adm,T_adm,U

def adm_pfr_ad(x_in, u_meas, theta, h_scale, l,n_steps):
    dl = l / (int(n_steps / 2))
    x = [x_in[0], x_in[4]]
    r = ode(PFR_adiabatic).set_integrator('vode', method='bdf')
    r.set_initial_value(x, 0).set_f_params(x_in, u_meas, theta, h_scale, l)
    l_n = np.linspace(0, l, int(n_steps / 2))
    ca_p = []
    T_p = []
    # for i in range(int(n/2)):
    while r.successful() and r.t < l:
        xf = r.integrate(r.t + dl)
        ca_p.append(xf[0])
        T_p.append(xf[1])

    cat = fsolve(adm_adiabatic, [ca_p[1:], T_p[1:]], args=(x_in, u_meas, theta, h_scale, l))
    ca_adm = cat[:int(len(cat) / 2)] / 1000
    T_adm = cat[int(len(cat) / 2):]

    tau = u_meas[1] #s
    TVR = u_meas[4]
    F_total = 21*1e-6/tau #m3/s
    c = theta[3]
    m = theta[4]
    n = theta[5]
    U = U_2(F_total, TVR, c,m,n)

    return ca_adm,T_adm,U

def para_gen_1(p_b, p_samples):
    #p_samples = 200
    n_phi = p_b.shape[0]  #

    # This code generates DoE by Latin hypercube sampling
    prng = 3
    lhs_points = lhs(n_phi, samples=p_samples, criterion='maximin', iterations=50, random_state=prng)
    paramset = np.zeros([p_samples, n_phi])
    for i in range(p_samples):
        for j in range(n_phi):
            paramset[i, j] = lhs_points[i, j] * (p_b[j, 1] - p_b[j, 0]) + p_b[j, 0]
    return paramset

def lhs_exp_cond(N_exp,prng):
    # Specification of experimental condition
    if N_exp ==1:
        l_bounds = [0.100,0.100, 5, 1., 100]
        u_bounds = [1.500, 1.500, 25, 10., 300]
        lhs_sampler = qmc.LatinHypercube(d=4)
        lhs_sample = lhs_sampler.random(n=1)
        lhs_point=qmc.scale(lhs_sample, l_bounds, u_bounds)
        lhs_actual=np.zeros([1, 5])
        lhs_actual[0,0] =lhs_point[0,0]
        lhs_actual[0,1] = lhs_point[0,1]
        lhs_actual[0,2] = lhs_point[0,2]
        lhs_actual[0,3] = lhs_point[0,3]
        lhs_actual[0, 4] = lhs_point[0, 4]
    else:
        control_bounds = [0.100,1.500, 0.100, 1.500, 5, 25, 1., 10., 100, 300]
        c_b = np.array(control_bounds).reshape(5, 2)

        #n_phi = c_b.shape[0]  #
        n_phi = 5
        # This code generates DoE by Latin hypercube sampling
        #prng = 3
        lhs_points = lhs(n_phi, samples=N_exp, criterion='maximin', iterations=50, random_state=prng)
        lhs_actual = np.zeros([N_exp, n_phi])
        for i in range(N_exp):
            lhs_actual[i, 0] = lhs_points[i, 0] * (c_b[0, 1] - c_b[0, 0]) + c_b[0, 0]
            lhs_actual[i, 1] = (lhs_points[i, 1] * (c_b[1, 1] - c_b[1, 0]) + c_b[1, 0])
            lhs_actual[i, 2] = lhs_points[i, 2] * (c_b[2, 1] - c_b[2, 0]) + c_b[2, 0]
            lhs_actual[i, 3] = lhs_points[i, 3] * (c_b[3, 1] - c_b[3, 0]) + c_b[3, 0]
            lhs_actual[i, 4] = round(lhs_points[i, 4] * (c_b[4, 1] - c_b[4, 0]) + c_b[4, 0])
    return lhs_actual

def para_gen_1_multi(p_b, p_samples,i):
    #p_samples = 200
    n_phi = p_b.shape[0]  #

    # This code generates DoE by Latin hypercube sampling
    prng = i
    lhs_points = lhs(n_phi, samples=p_samples, criterion='maximin', iterations=50, random_state=prng)
    paramset = np.zeros([p_samples, n_phi])
    for i in range(p_samples):
        for j in range(n_phi):
            paramset[i, j] = lhs_points[i, j] * (p_b[j, 1] - p_b[j, 0]) + p_b[j, 0]
    return paramset

def TVR_ADM_model(para_KU,input_cond):
    N_exp = len(input_cond)
    rho_a = 1.05 #g/cm3
    rho_b = 0.98 #g/cm3
    mwa = 106.12  # g/mol
    mwb = 107.15  # g/mol

    conc_st_a= rho_a/mwa *1000 #mol/dm3
    conc_st_b= rho_b/mwb *1000 #mol/dm3

    k_cond = 0.2 #W/(mK)
    cpm = 2534  # J/(kg.K)
    rho_m = 792  # kg/m^3

    pm = ([para_KU[0],para_KU[1]])
    n_var = 5
    x_meas = np.zeros([N_exp, n_var, 5])

    cstr_number_data = np.zeros([N_exp])
    cstr_number_gp = np.zeros([N_exp])
    u_meas = np.zeros([N_exp, 4]) # control variables jacket temperature and TVR spped
    t = np.zeros([N_exp])
    react_temp_out = np.zeros([N_exp])
    react_temp_out_2 = np.zeros([N_exp])
    TR1 = np.zeros([N_exp])
    T_atm = np.zeros([N_exp])
    TR2 = np.zeros([N_exp])
    TR3 = np.zeros([N_exp])
    TR4 = np.zeros([N_exp])
    conc_data = np.zeros([N_exp])
    ca_data = np.zeros([N_exp])
    cb_data = np.zeros([N_exp])
    conversion_data = np.zeros([N_exp])
    conversion_cstr = np.zeros([N_exp,2])
    c_std = np.zeros([N_exp])
    tr4_std = np.zeros([N_exp])
    tr3_std = np.zeros([N_exp])
    tr2_std = np.zeros([N_exp])
    conv_std = np.zeros([N_exp])
    F_123 = np.zeros([N_exp,3])
    ca_inlet = np.zeros([N_exp])
    cb_inlet = np.zeros([N_exp])


    h_scale = 1000

    for i in range(N_exp):
        conversion_data[i] = input_cond[i, 5] # conversion data
        t[i]            = input_cond[i, 3] # residence time
        react_temp_out[i] = input_cond[i, 10]
        react_temp_out_2[i] = input_cond[i, 9]
        TR1[i] = input_cond[i, 7]
        T_atm[i] = input_cond[i, 6]
        TR2[i] = input_cond[i, 8]
        TR3[i] = input_cond[i, 9]
        TR4[i] = input_cond[i, 10]
        u_meas[i,0] = input_cond[i, 2]
        u_meas[i, 1] = input_cond[i, 4]
        u_meas[i, 2] = input_cond[i, 3]

        if input_cond[i, 0] <= input_cond[i, 1]:
            conc_data[i] = (1 - conversion_data[i]) * input_cond[i, 0]
        else:
            conc_data[i] = input_cond[i, 0] - (conversion_data[i] * input_cond[i, 1])
        if input_cond[i, 0] <= input_cond[i, 1]:
            conc_data[i] = (1 - conversion_data[i]) * input_cond[i, 0]
        else:
            conc_data[i] = input_cond[i, 0] - (conversion_data[i] * input_cond[i, 1])
        c_std[i] = 0.012
        tr4_std[i] = 1.7
        tr3_std[i] = 0.6
        tr2_std[i] = 0.6
        conv_std[i] = 1.2
        F_123[i, 0] = input_cond[i, 0] / conc_st_a * 21e-6 / t[i]
        F_123[i, 1] = input_cond[i, 0] / conc_st_b * 21e-6 / t[i]
        F_123[i, 2] = 21e-6 / t[i] - F_123[i, 0] - F_123[i, 1]
        ca_inlet[i] = F_123[i,0]/(F_123[i,0]+F_123[i,2])*conc_st_a
        cb_inlet[i] = conc_st_b

    l0 = 1/30
    la=1/3
    lb=2/3*9/10#2/3*9/10
    lc=2/3*1/10 #2/3*1/10

    for i in range(N_exp):
        x_in = [input_cond[i, 0] * 1000, input_cond[i, 1] * 1000, 0, 0, input_cond[i, 7]]
        Tc = input_cond[i, 2]
        tau = input_cond[i, 3] * 60  # s
        u = input_cond[i, 11]  # 8.15e-4
        Da = input_cond[i, 12]  # 4.7e-5
        omega = input_cond[i,4]

        u_meas = [Tc, tau, u, Da, omega, input_cond[i, 6]]

        kref = pm[0]/1000 #2.5e-5
        Ea = pm[0]*1000#39200
        lamda_a = k_cond+rho_m*cpm*Da
        n_steps = 20
        L_R = 0.3
        x_meas[i,:,:] = sub_response_fn(ca_inlet[i],input_cond[i],n_var,u_meas, kref, Ea, lamda_a, para_KU[2:], h_scale, L_R, l0,la,lb,lc, n_steps,F_123[i],conc_st_b)

    for i in range(N_exp):
        if x_meas[i, 0, 2] <= x_meas[i, 1,2]:
            conversion_cstr[i,0] = (x_meas[i, 0, 2] - x_meas[i, 0, 3]) / x_meas[i, 0, 2] * 100
            conversion_cstr[ i, 1] = (x_meas[ i, 0, 2] - x_meas[ i, 0, 4]) / x_meas[ i, 0, 2] * 100
            ca_data[i] = input_cond[i, 0]* (1 - conversion_data[i]/100)
            cb_data[i] = input_cond[i, 1] - (input_cond[i, 0]-ca_data[i])
        else:
            conversion_cstr[i,0] = (x_meas[i, 1, 2] - x_meas[i, 1, 3]) / x_meas[i, 1, 2] * 100
            conversion_cstr[ i, 1] = (x_meas[ i, 1, 2] - x_meas[ i, 1, 4]) / x_meas[ i, 1, 2] * 100
            cb_data[i] = input_cond[i, 1]* (1 - conversion_data[i]/100)
            ca_data[i] = input_cond[i, 0] - (input_cond[i, 1]-cb_data[i])

    conversion_cstr[conversion_cstr > 100] = 100

    mle = 0

    for i in range(N_exp):
        mle += maximum_likelihood_est(x_meas[i,0,4], ca_data[i], x_meas[i,4,4], react_temp_out[i], c_std[i],tr4_std[i])

    return mle
def maximum_likelihood_est( ca, ca_data, T4, T4_data, sigma_ca,sigma_T):
    """
    This function compute the MLE for a given set of equations
    """
    MLE =   0.5 * ((ca - ca_data) ** 2 / (sigma_ca) ** 2 +(T4 - T4_data) ** 2 / (sigma_T) ** 2)
    return MLE

def compute_fim(ca_inlet, input_cond, n_var, u_meas, kref, Ea, lamda_a, para_U_in, h_scale, L_R, l0, la, lb, lc, n_steps,
                F_123, conc_st_b, ca_sd, TR4_sd):
    x_unperturbed = sub_response_fn_2(ca_inlet, input_cond, n_var, u_meas, kref, Ea, lamda_a, para_U_in, h_scale, L_R, l0,
                                      la, lb, lc, n_steps, F_123, conc_st_b)
    para_U = np.hstack([kref, Ea,para_U_in])
    n_param = len(para_U)
    n_var = len(x_unperturbed)
    dx_dparam = np.zeros((n_var, n_param))
    sigma = np.array([ca_sd, TR4_sd])
    param_eps = 0.01
    for i in range(n_param):
        param_perturbed = para_U.copy()
        param_perturbed[i] += param_eps * param_perturbed[i]
        x_perturbed = sub_response_fn_2(ca_inlet, input_cond, n_var, u_meas, param_perturbed[0], param_perturbed[1], lamda_a, param_perturbed[2:], h_scale, L_R,
                                        l0, la, lb, lc, n_steps, F_123, conc_st_b)
        dx_dparam[:, i] = (x_perturbed - x_unperturbed) / param_eps * param_perturbed[i]

    FIM = dx_dparam.T @ np.linalg.inv(np.diag(np.square(sigma[:]))) @ dx_dparam
    return FIM

def sub_response_fn_2(ca_inlet,input_cond,n_var,u_meas,kref, Ea, lamda_a, para_U, h_scale, L_R, l0,la,lb,lc, n_steps,F_123,conc_st_b):
    theta1 = [kref, Ea, lamda_a, para_U[0], para_U[1], para_U[2], para_U[0]]
    theta2 = [kref, Ea, lamda_a, para_U[3], para_U[4], para_U[5], para_U[2]]
    theta3 = [kref, Ea, lamda_a, para_U[6], para_U[7], para_U[8], para_U[3]]
    theta4 = [kref, Ea, lamda_a, para_U[9], para_U[10], para_U[11], para_U[3]]

    x_meas = np.zeros([ 5, 5])
    x_meas[0, 0] = ca_inlet
    x_meas[4, 0] = input_cond[7]  # temperature
    x_in_atm = [x_meas[0, 0] * 1000, 0, 0, 0, x_meas[ 4, 0]]
    ca_adm_nr_atm, T_adm_nr_atm,_ = adm_pfr_nr_atm(x_in_atm, u_meas, theta1, h_scale, L_R * l0, n_steps)
    x_meas[ 0, 1] = ca_adm_nr_atm[-1]
    x_meas[ 4, 1] = T_adm_nr_atm[-1]
    x_in_nr = [x_meas[ 0, 1] * 1000, 0, 0, 0, x_meas[ 4, 1]]
    ca_adm_nr, T_adm_nr,_ = adm_pfr_nr(x_in_nr, u_meas, theta2, h_scale, L_R * la, n_steps)
    x_meas[ 4, 2] = T_adm_nr[-1]
    x_meas[ 0, 2] = x_meas[ 0, 0] * ((F_123[ 0] + F_123[ 2]) / (F_123[ 0] + F_123[ 1] + F_123[ 2]))
    x_meas[ 1, 2] = conc_st_b * (
            (F_123[ 1]) / (F_123[ 0] + F_123[ 1] + F_123[ 2]))
    x_in_na = [x_meas[ 0, 2] * 1000, x_meas[ 1, 2] * 1000, x_meas[ 2, 1] * 1000, x_meas[ 3, 1] * 1000,
               x_meas[ 4, 1]]
    ca_adm_na, T_adm_na,_ = adm_pfr_na(x_in_na, u_meas, theta3, h_scale, L_R * lb, n_steps)
    x_meas[ 0, 3] = ca_adm_na[-1]
    x_meas[ 1, 3] = x_meas[ 1, 2] - (x_meas[ 0, 2] - ca_adm_na[-1])
    x_meas[ 2, 3] = x_meas[ 2, 2] + (x_meas[ 0, 2] - ca_adm_na[-1])
    x_meas[ 3, 3] = x_meas[ 3, 2] + (x_meas[ 0, 2] - ca_adm_na[-1])
    x_meas[ 4, 3] = T_adm_na[-1]
    x_in_ad = [x_meas[ 0, 3] * 1000, x_meas[ 1, 3] * 1000, x_meas[ 2, 3] * 1000, x_meas[ 3, 3] * 1000,
               x_meas[ 4, 3]]
    ca_adm_ad, T_adm_ad,_ = adm_pfr_ad(x_in_ad, u_meas, theta4, h_scale, L_R * lc, n_steps)
    x_meas[ 0, 4] = ca_adm_ad[-1]
    x_meas[ 1, 4] = x_meas[ 1, 3] - (x_meas[ 0, 3] - ca_adm_ad[-1])
    x_meas[ 2, 4] = x_meas[ 2, 3] + (x_meas[ 0, 3] - ca_adm_ad[-1])
    x_meas[ 3, 4] = x_meas[ 3, 3] + (x_meas[ 0, 3] - ca_adm_ad[-1])
    x_meas[ 4, 4] = T_adm_ad[-1]
    return  np.array([x_meas[0,4],x_meas[4,4]])

def sub_response_fn(ca_inlet,input_cond,n_var,u_meas,kref, Ea, lamda_a, para_U, h_scale, L_R, l0,la,lb,lc, n_steps,F_123,conc_st_b):
    theta1 = [kref, Ea, lamda_a, para_U[0], para_U[1], para_U[2], para_U[0]]
    theta2 = [kref, Ea, lamda_a, para_U[3], para_U[4], para_U[5], para_U[2]]
    theta3 = [kref, Ea, lamda_a, para_U[6], para_U[7], para_U[8], para_U[3]]
    theta4 = [kref, Ea, lamda_a, para_U[9], para_U[10], para_U[11], para_U[3]]

    x_meas = np.zeros([ 5, 5])
    x_meas[0, 0] = ca_inlet
    x_meas[4, 0] = input_cond[7]  # temperature
    x_in_atm = [x_meas[0, 0] * 1000, 0, 0, 0, x_meas[ 4, 0]]
    ca_adm_nr_atm, T_adm_nr_atm,_ = adm_pfr_nr_atm(x_in_atm, u_meas, theta1, h_scale, L_R * l0, n_steps)
    x_meas[ 0, 1] = ca_adm_nr_atm[-1]
    x_meas[ 4, 1] = T_adm_nr_atm[-1]
    x_in_nr = [x_meas[ 0, 1] * 1000, 0, 0, 0, x_meas[ 4, 1]]
    ca_adm_nr, T_adm_nr,_ = adm_pfr_nr(x_in_nr, u_meas, theta2, h_scale, L_R * la, n_steps)
    x_meas[ 4, 2] = T_adm_nr[-1]
    x_meas[ 0, 2] = x_meas[ 0, 0] * ((F_123[ 0] + F_123[ 2]) / (F_123[ 0] + F_123[ 1] + F_123[ 2]))
    x_meas[ 1, 2] = conc_st_b * (
            (F_123[ 1]) / (F_123[ 0] + F_123[ 1] + F_123[ 2]))
    x_in_na = [x_meas[ 0, 2] * 1000, x_meas[ 1, 2] * 1000, x_meas[ 2, 1] * 1000, x_meas[ 3, 1] * 1000,
               x_meas[ 4, 1]]
    ca_adm_na, T_adm_na,_ = adm_pfr_na(x_in_na, u_meas, theta3, h_scale, L_R * lb, n_steps)
    x_meas[ 0, 3] = ca_adm_na[-1]
    x_meas[ 1, 3] = x_meas[ 1, 2] - (x_meas[ 0, 2] - ca_adm_na[-1])
    x_meas[ 2, 3] = x_meas[ 2, 2] + (x_meas[ 0, 2] - ca_adm_na[-1])
    x_meas[ 3, 3] = x_meas[ 3, 2] + (x_meas[ 0, 2] - ca_adm_na[-1])
    x_meas[ 4, 3] = T_adm_na[-1]
    x_in_ad = [x_meas[ 0, 3] * 1000, x_meas[ 1, 3] * 1000, x_meas[ 2, 3] * 1000, x_meas[ 3, 3] * 1000,
               x_meas[ 4, 3]]
    ca_adm_ad, T_adm_ad,_ = adm_pfr_ad(x_in_ad, u_meas, theta4, h_scale, L_R * lc, n_steps)
    x_meas[ 0, 4] = ca_adm_ad[-1]
    x_meas[ 1, 4] = x_meas[ 1, 3] - (x_meas[ 0, 3] - ca_adm_ad[-1])
    x_meas[ 2, 4] = x_meas[ 2, 3] + (x_meas[ 0, 3] - ca_adm_ad[-1])
    x_meas[ 3, 4] = x_meas[ 3, 3] + (x_meas[ 0, 3] - ca_adm_ad[-1])
    x_meas[ 4, 4] = T_adm_ad[-1]
    return  x_meas

def PB_prediction_var(input_cond,para_KU, z_sd):
    p_samples=10

    #material parameters
    rho_a = 1.05  # g/cm3
    rho_b = 0.98  # g/cm3
    mwa = 106.12  # g/mol
    mwb = 107.15  # g/mol

    conc_st_a = rho_a / mwa * 1000  # mol/dm3
    conc_st_b = rho_b / mwb * 1000  # mol/dm3
    k_cond = 0.2  # W/(mK)
    cpm = 2534  # J/(kg.K)
    rho_m = 792  # kg/m^3

    pm = ([para_KU[0],para_KU[1]])
    ps = ([z_sd[0,0],z_sd[1,1]])
    bounds_kp = [(pm[0] - 2*ps[0], pm[0] + 2*ps[0]),
                (pm[1] - 2*ps[1], pm[1] + 2*ps[1])]

    cz_1, mz_1,nz_1 = para_KU[2],para_KU[3],para_KU[4]
    cz_2, mz_2,nz_2 = para_KU[8],para_KU[9],para_KU[10]
    cz_3, mz_3,nz_3 = para_KU[11],para_KU[12],para_KU[13]

    cz_sd_1, mz_sd_1,nz_sd_1 = z_sd[2,2],z_sd[3,3],z_sd[4,4]
    cz_sd_2, mz_sd_2,nz_sd_2 = z_sd[8,8],z_sd[9,9],z_sd[10,10]
    cz_sd_3, mz_sd_3,nz_sd_3 = z_sd[11,11],z_sd[12,12],z_sd[13,13]

    bounds_z = [(cz_1 - 2*cz_sd_1, cz_1 + 2*cz_sd_1),
                (mz_1 - 2*mz_sd_1, mz_1 + 2*mz_sd_1),
                (nz_1 - 2*nz_sd_1, nz_1 + 2*nz_sd_1),
                (cz_1 - 2 * cz_sd_1, cz_1 + 2 * cz_sd_1),
                (mz_1 - 2 * mz_sd_1, mz_1 + 2 * mz_sd_1),
                (nz_1 - 2 * nz_sd_1, nz_1 + 2 * nz_sd_1),
                (cz_2 - 2*cz_sd_2, cz_2 + 2*cz_sd_2),
                (mz_2 - 2*mz_sd_2, mz_2 + 2*mz_sd_2),
                (nz_2 - 2*nz_sd_2, nz_2 + 2*nz_sd_2),
                (cz_3 - 2 * cz_sd_3, cz_3 + 2 * cz_sd_3),
                (mz_3 - 2 * mz_sd_3, mz_3 + 2 * mz_sd_3),
                (nz_3 - 2 * nz_sd_3, nz_3 + 2 * nz_sd_3)
                ]

    k_p= para_gen_1(np.array(bounds_kp), p_samples)
    para_U_z = para_gen_1(np.array(bounds_z), p_samples)
    N_p = p_samples

    n_var = 5
    x_meas = np.zeros([N_p, n_var, 5])

    U_zone_1 = np.zeros([N_p])
    U_zone_2 = np.zeros([N_p])
    U_zone_3 = np.zeros([N_p])
    U_zone_4 = np.zeros([N_p])

    u_meas = np.zeros([4]) # control variables jacket temperature and TVR spped
    F_123 = np.zeros([3])
    h_scale = 1000

    for j in range(N_p):
        conversion_data= input_cond[5] # conversion data
        x_meas[j, 4, 0] = input_cond[ 7] # temperature
        t            = input_cond[3] # residence time
        u_meas[0] = input_cond[ 2]
        u_meas[ 1] = input_cond[ 4]
        u_meas[ 2] = input_cond[ 3]

        F_123[0] = input_cond[ 0]/conc_st_a*21e-6/t
        F_123[ 1] = input_cond[ 0]/conc_st_b*21e-6/t
        F_123[ 2] = 21e-6/t - F_123[0] - F_123[1]
        ca_inlet = F_123[0]/(F_123[0]+F_123[2])*conc_st_a

    l0 =9/300# 1/30 #9/300
    la=91/300#1/3 #91/300
    lb=191/300#2/3*9/10 #191/300
    lc=9/300#2/3*1/10#9/300 #2/3*1/10
    n_steps = 10
    L_R = 0.3

    for j in range(N_p):
        Tc = input_cond[2]
        tau = input_cond[3]*60 #s
        u = input_cond[11]#8.15e-4
        Da = input_cond[12]#4.7e-5
        omega = input_cond[4]

        u_meas = [Tc, tau, u, Da, omega,input_cond[6]]

        kref = k_p[j,0]/1000 #2.5e-5
        Ea = k_p[j,1]*1000#39200
        lamda_a = (k_cond+rho_m*cpm*Da)/1
        theta1 =[kref,Ea,lamda_a,para_U_z[j,0], para_U_z[j,1], para_U_z[j,2]]
        theta2 = [kref, Ea, lamda_a, para_U_z[j, 3], para_U_z[j, 4], para_U_z[j, 5]]
        theta3 = [kref, Ea, lamda_a, para_U_z[j, 6], para_U_z[j, 7], para_U_z[j, 8]]
        theta4 = [kref, Ea, lamda_a, para_U_z[j, 9], para_U_z[j, 10], para_U_z[j, 11]]

        x_meas[j,0,0] = ca_inlet
        x_in_atm = [x_meas[j,0,0] * 1000, 0, 0, 0, x_meas[j,4,0]]
        ca_adm_nr_atm,T_adm_nr_atm,U_zone_1[j] = adm_pfr_nr_atm(x_in_atm, u_meas, theta1, h_scale, L_R*l0, n_steps)
        x_meas[j, 0, 1] = ca_adm_nr_atm[-1]
        x_meas[j, 4, 1] = T_adm_nr_atm[-1]
        x_in_nr=[x_meas[j, 0, 1]*1000, 0, 0,0,x_meas[j, 4, 1]]
        ca_adm_nr,T_adm_nr,U_zone_2[j] = adm_pfr_nr(x_in_nr, u_meas, theta2, h_scale, L_R*la, n_steps)
        x_meas[j, 4, 2] = T_adm_nr[-1]
        x_meas[j, 0, 2] = x_meas[j,0,0]* ((F_123[0]+F_123[2])/(F_123[0]+F_123[1]+F_123[2]))
        x_meas[j, 1, 2] = conc_st_b * (
                    (F_123[ 1]) / (F_123[ 0] + F_123[ 1] + F_123[ 2]))
        x_in_na = [x_meas[j, 0, 2]*1000, x_meas[j, 1, 2]*1000, x_meas[j, 2, 1]*1000,x_meas[j, 3, 1]*1000,x_meas[j, 4, 1]]
        ca_adm_na,T_adm_na,U_zone_3[j] = adm_pfr_na(x_in_na, u_meas, theta3, h_scale, L_R*lb, n_steps)
        x_meas[j, 0, 3] =ca_adm_na[-1]
        x_meas[j, 1, 3] =x_meas[j, 1, 2] - (x_meas[j, 0, 2]- ca_adm_na[-1])
        x_meas[j, 2, 3] = x_meas[j, 2, 2]+(x_meas[j, 0, 2] - ca_adm_na[-1])
        x_meas[j, 3, 3] =  x_meas[j, 3, 2]+ (x_meas[j, 0, 2] - ca_adm_na[-1])
        x_meas[j, 4, 3] = T_adm_na[-1]
        x_in_ad = [x_meas[j, 0, 3]*1000, x_meas[j, 1, 3]*1000, x_meas[j, 2, 3]*1000,x_meas[j, 3, 3]*1000,x_meas[j, 4, 3]]
        ca_adm_ad,T_adm_ad,U_zone_4[j] = adm_pfr_ad(x_in_ad, u_meas, theta4, h_scale, L_R*lc, n_steps)
        x_meas[j, 0, 4] =ca_adm_ad[-1]
        x_meas[j, 1, 4] =x_meas[j, 1, 3] - (x_meas[j, 0, 3]- ca_adm_ad[-1])
        x_meas[j, 2, 4] = x_meas[j, 2, 3]+ (x_meas[j, 0, 3] - ca_adm_ad[-1])
        x_meas[j, 3, 4] = x_meas[j, 3, 3]+ (x_meas[j, 0, 3] - ca_adm_ad[-1])
        x_meas[j, 4, 4] = T_adm_ad[-1]

    return np.std(x_meas[:,0,-1]), np.mean(x_meas[:,0,-1])

def GP_var(X_new,model):

    mean, std = model.predict(X_new.reshape(1,-1), return_std=True)

    return std, mean

def parameter_estim(input_cond):
    cz_1, mz_1, nz_1 = 0.425, 1.203, 0.119
    cz_2, mz_2, nz_2 = 0.141, 1.494, 0.222
    cz_3, mz_3, nz_3 = 0.293, 0.851, 0.217
    pm = ([0.025, 39.2])
    ps = ([0.025 * 0.037, 39.2 * 0.109])

    cz_sd_1, mz_sd_1, nz_sd_1 = 0.037, 0.048, 0.037
    cz_sd_2, mz_sd_2, nz_sd_2 = 0.037, 0.185, 0.144
    cz_sd_3, mz_sd_3, nz_sd_3 = 0.022, 0.041, 0.032
    CI = 10
    CI_k = 2
    bounds_z = [(pm[0] - CI_k * ps[0], pm[0] + CI_k * ps[0]),
                (pm[1] - CI_k * ps[1], pm[1] + CI_k * ps[1]),
                (cz_1 - CI * cz_sd_1, cz_1 + CI * cz_sd_1),
                (mz_1 - CI * mz_sd_1, mz_1 + CI * mz_sd_1),
                (nz_1 - CI * nz_sd_1, nz_1 + CI * nz_sd_1),
                (cz_1 - CI * cz_sd_1, cz_1 + CI * cz_sd_1),
                (mz_1 - CI * mz_sd_1, mz_1 + CI * mz_sd_1),
                (nz_1 - CI * nz_sd_1, nz_1 + CI * nz_sd_1),
                (cz_2 - CI * cz_sd_2, cz_2 + CI * cz_sd_2),
                (mz_2 - CI * mz_sd_2, mz_2 + CI * mz_sd_2),
                (nz_2 - CI * nz_sd_2, nz_2 + CI * nz_sd_2),
                (cz_3 - CI * cz_sd_3, cz_3 + CI * cz_sd_3),
                (mz_3 - CI * mz_sd_3, mz_3 + CI * mz_sd_3),
                (nz_3 - CI * nz_sd_3, nz_3 + CI * nz_sd_3)
                ]
    param_sd = ([ps[0], ps[1],
                 cz_sd_1, mz_sd_1, nz_sd_1,
                 cz_sd_1, mz_sd_1, nz_sd_1,
                 cz_sd_2, mz_sd_2, nz_sd_2,
                 cz_sd_3, mz_sd_3, nz_sd_3])

    p_cov0 = np.diag(np.square(param_sd[:]))
    guess_samples = 10
    n_prng = 3
    res_mle = np.zeros([n_prng, len(bounds_z) + 1])
    for pp in range(n_prng):
        multi_para= para_gen_1_multi(np.array(bounds_z), guess_samples,pp)
        mle_up = np.zeros([guess_samples,len(bounds_z)+1])
        for kk in range(guess_samples):
            mle_up[kk, :-1] = multi_para[kk]
            mle_up[kk,-1] = TVR_ADM_model(multi_para[kk],input_cond)
        df = pd.DataFrame(mle_up, columns =['k_1','k_2','cz_1','mz_1','nz_1', 'cz_2','mz_2','nz_2', 'cz_3','mz_3','nz_3', 'cz_4','mz_4','nz_4', 'mle']).sort_values('mle',ascending=True)
        res = minimize(TVR_ADM_model, df.values[0][:-1],args=(input_cond), bounds=bounds_z, method="L-BFGS-B",options={'gtol': 1e-6, 'disp': True})
        res_mle[pp,:-1] = res.x
        res_mle[pp, -1] = res.fun
    df2 = pd.DataFrame(res_mle, columns =['k_1','k_2','cz_1','mz_1','nz_1', 'cz_2','mz_2','nz_2', 'cz_3','mz_3','nz_3', 'cz_4','mz_4','nz_4', 'mle']).sort_values('mle',ascending=True)

    para_KU = df2.values[0][:-1]

    N_exp = len(input_cond)
    rho_a = 1.05  # g/cm3
    rho_b = 0.98  # g/cm3
    mwa = 106.12  # g/mol
    mwb = 107.15  # g/mol

    conc_st_a = rho_a / mwa * 1000  # mol/dm3
    conc_st_b = rho_b / mwb * 1000  # mol/dm3

    k_cond = 0.2  # W/(mK)
    cpm = 2534  # J/(kg.K)
    rho_m = 792  # kg/m^3

    n_var = 5

    u_meas = np.zeros([N_exp, 4])  # control variables jacket temperature and TVR spped
    t = np.zeros([N_exp])
    react_temp_out = np.zeros([N_exp])
    react_temp_out_2 = np.zeros([N_exp])
    TR1 = np.zeros([N_exp])
    T_atm = np.zeros([N_exp])
    TR2 = np.zeros([N_exp])
    TR3 = np.zeros([N_exp])
    TR4 = np.zeros([N_exp])
    conc_data = np.zeros([N_exp])
    conversion_data = np.zeros([N_exp])
    c_std = np.zeros([N_exp])
    tr4_std = np.zeros([N_exp])
    tr3_std = np.zeros([N_exp])
    tr2_std = np.zeros([N_exp])
    conv_std = np.zeros([N_exp])
    F_123 = np.zeros([N_exp, 3])
    ca_inlet = np.zeros([N_exp])
    cb_inlet = np.zeros([N_exp])

    h_scale = 1000
    for i in range(N_exp):
        conversion_data[i] = input_cond[i, 5]  # conversion data
        t[i] = input_cond[i, 3]  # residence time
        react_temp_out[i] = input_cond[i, 10]
        react_temp_out_2[i] = input_cond[i, 9]
        TR1[i] = input_cond[i, 7]
        T_atm[i] = input_cond[i, 6]
        TR2[i] = input_cond[i, 8]
        TR3[i] = input_cond[i, 9]
        TR4[i] = input_cond[i, 10]
        u_meas[i, 0] = input_cond[i, 2]
        u_meas[i, 1] = input_cond[i, 4]
        u_meas[i, 2] = input_cond[i, 3]

        if input_cond[i, 0] <= input_cond[i, 1]:
            conc_data[i] = (1 - conversion_data[i]) * input_cond[i, 0]
        else:
            conc_data[i] = input_cond[i, 0] - (conversion_data[i] * input_cond[i, 1])
        c_std[i] = 0.012
        tr4_std[i] = 1.7
        tr3_std[i] = 0.6
        tr2_std[i] = 0.6
        conv_std[i] = 1.2
        F_123[i, 0] = input_cond[i, 0] / conc_st_a * 21e-6 / t[i]
        F_123[i, 1] = input_cond[i, 0] / conc_st_b * 21e-6 / t[i]
        F_123[i, 2] = 21e-6 / t[i] - F_123[i, 0] - F_123[i, 1]
        ca_inlet[i] = F_123[i, 0] / (F_123[i, 0] + F_123[i, 2]) * conc_st_a
        cb_inlet[i] = conc_st_b

    l0 = 1 / 30
    la = 1 / 3
    lb = 2 / 3 * 9 / 10  # 2/3*9/10
    lc = 2 / 3 * 1 / 10  # 2/3*1/10

    cFIM = 0
    for i in range(N_exp):
        x_in = [input_cond[i, 0] * 1000, input_cond[i, 1] * 1000, 0, 0, input_cond[i, 7]]
        Tc = input_cond[i, 2]
        tau = input_cond[i, 3] * 60  # s
        u = input_cond[i, 11]  # 8.15e-4
        Da = input_cond[i, 12]  # 4.7e-5
        omega = input_cond[i, 4]

        u_meas = [Tc, tau, u, Da, omega, input_cond[i, 6]]

        kref = para_KU[0] / 1000  # 2.5e-5
        Ea = para_KU[1] * 1000  # 39200
        lamda_a = k_cond + rho_m * cpm * Da
        n_steps = 20
        x_out = sp.symbols(f'x0:{n_steps}')
        L_R = 0.3

        cFIM += compute_fim(ca_inlet[i], input_cond[i], n_var, u_meas, kref, Ea, lamda_a, para_KU[2:], h_scale, L_R, l0,
                            la, lb, lc, n_steps,
                            F_123[i], conc_st_b, c_std[i], tr4_std[i])


    p_cov_post_inv = np.linalg.inv(p_cov0) + cFIM
    p_cov_post = np.linalg.inv(p_cov_post_inv)
    z_sd = np.sqrt(p_cov_post)
    return para_KU, z_sd

def close_loop(output_file,input_file):

    rho_a = 1.05  # g/cm3
    rho_b = 0.98  # g/cm3
    mwa = 106.12  # g/mol
    mwb = 107.15  # g/mol

    conc_st_a = rho_a / mwa * 1000  # mol/dm3
    conc_st_b = rho_b / mwb * 1000  # mol/dm3

    k_cond = 0.2  # W/(mK)
    cpm = 2534  # J/(kg.K)
    rho_m = 792  # kg/m^3

    p_samples = 10
    #parser = ArgumentParser()
    #parser.add_argument("input_file_path", help="Input CSV file", type=FileType('r'), nargs=1)

    #data = pd.read_csv(parser.parse_args().output_file[0])

    data = pd.read_csv(output_file)
    output_data = data[
        ['Benzaldehyde FR', 'Benzylamine FR', 'Methanol FR', 'Temp', 'Rotor Speed','CIF', 'TF','TR1', 'TR2','TR3', 'TR4','TJ1',
         'TJ2','RT', 'Rotor (W)','Huber (W)','Conversion']].values

    input_cond=np.zeros((len(output_data),13))

    for ic in range(len(output_data)):
        input_cond[ic,0] = output_data[ic,0]/(output_data[ic,0]+output_data[ic,1]+output_data[ic,2])*conc_st_a
        input_cond[ic, 1] = output_data[ic, 1] / (
                    output_data[ic, 0] + output_data[ic, 1] + output_data[ic, 2]) * conc_st_b
        input_cond[ic, 2] = output_data[ic, 3]
        input_cond[ic, 3] = 1/((output_data[ic,0]+output_data[ic,1]+output_data[ic,2])/21e3/60)
        input_cond[ic, 4] = output_data[ic, 4]
        input_cond[ic, 5] = output_data[ic, 16]
        input_cond[ic, 6] = output_data[ic, 13]
        input_cond[ic, 7] = output_data[ic, 7]
        input_cond[ic, 8] = output_data[ic, 8]
        input_cond[ic, 9] = output_data[ic, 9]
        input_cond[ic, 10] = output_data[ic, 10]
        input_cond[ic, 11] = 8.15e-4
        input_cond[ic, 12] = 4.7e-5

    para_KU, z_sd=parameter_estim(input_cond)
    pm = ([para_KU[0],para_KU[1]])
    ps = ([z_sd[0,0],z_sd[1,1]])
    bounds_kp = [(pm[0] - 2*ps[0], pm[0] + 2*ps[0]),
                (pm[1] - 2*ps[1], pm[1] + 2*ps[1])]

    cz_1, mz_1,nz_1 = para_KU[2],para_KU[3],para_KU[4]
    cz_2, mz_2,nz_2 = para_KU[8],para_KU[9],para_KU[10]
    cz_3, mz_3,nz_3 = para_KU[11],para_KU[12],para_KU[13]

    cz_sd_1, mz_sd_1,nz_sd_1 = z_sd[2,2],z_sd[3,3],z_sd[4,4]
    cz_sd_2, mz_sd_2,nz_sd_2 = z_sd[8,8],z_sd[9,9],z_sd[10,10]
    cz_sd_3, mz_sd_3,nz_sd_3 = z_sd[11,11],z_sd[12,12],z_sd[13,13]
    N_exp = len(input_cond)

    bounds_z = [(cz_1 - 2*cz_sd_1, cz_1 + 2*cz_sd_1),
                (mz_1 - 2*mz_sd_1, mz_1 + 2*mz_sd_1),
                (nz_1 - 2*nz_sd_1, nz_1 + 2*nz_sd_1),
                (cz_1 - 2 * cz_sd_1, cz_1 + 2 * cz_sd_1),
                (mz_1 - 2 * mz_sd_1, mz_1 + 2 * mz_sd_1),
                (nz_1 - 2 * nz_sd_1, nz_1 + 2 * nz_sd_1),
                (cz_2 - 2*cz_sd_2, cz_2 + 2*cz_sd_2),
                (mz_2 - 2*mz_sd_2, mz_2 + 2*mz_sd_2),
                (nz_2 - 2*nz_sd_2, nz_2 + 2*nz_sd_2),
                (cz_3 - 2 * cz_sd_3, cz_3 + 2 * cz_sd_3),
                (mz_3 - 2 * mz_sd_3, mz_3 + 2 * mz_sd_3),
                (nz_3 - 2 * nz_sd_3, nz_3 + 2 * nz_sd_3)
                ]

    k_p= para_gen_1(np.array(bounds_kp), p_samples)
    para_U_z = para_gen_1(np.array(bounds_z), p_samples)
    N_p = p_samples

    n_var = 5
    x_meas = np.zeros([N_p,N_exp, n_var, 5])

    U_zone_1 = np.zeros([N_p,N_exp])
    U_zone_2 = np.zeros([N_p,N_exp])
    U_zone_3 = np.zeros([N_p,N_exp])
    U_zone_4 = np.zeros([N_p,N_exp])

    u_meas = np.zeros([N_exp, 4]) # control variables jacket temperature and TVR spped
    t = np.zeros([N_exp])
    react_temp_out = np.zeros([N_exp])
    react_temp_out_2 = np.zeros([N_exp])
    TR1 = np.zeros([N_exp])
    T_atm = np.zeros([N_exp])
    TR2 = np.zeros([N_exp])
    TR3 = np.zeros([N_exp])
    TR4 = np.zeros([N_exp])
    conc_data = np.zeros([N_exp])
    ca_data = np.zeros([N_exp])
    cb_data = np.zeros([N_exp])
    conversion_data = np.zeros([N_exp])
    conversion_cstr = np.zeros([N_p,N_exp,2])
    c_std = np.zeros([N_exp])
    tr4_std = np.zeros([N_exp])
    tr3_std = np.zeros([N_exp])
    tr2_std = np.zeros([N_exp])
    conv_std = np.zeros([N_exp])
    F_123 = np.zeros([N_exp,3])
    ca_inlet = np.zeros([N_exp])
    cb_inlet = np.zeros([N_exp])
    conv_sd = np.zeros([N_exp])

    h_scale = 1000

    for j in range(N_p):
        for i in range(N_exp):
            conversion_data[i] = input_cond[i, 5] # conversion data
            x_meas[j,i, 4, 0] = input_cond[i, 7] # temperature
            t[i]            = input_cond[i, 3] # residence time
            react_temp_out[i] = input_cond[i, 10]
            react_temp_out_2[i] = input_cond[i, 9]
            TR1[i] = input_cond[i, 7]
            T_atm[i] = input_cond[i, 6]
            TR2[i] = input_cond[i, 8]
            TR3[i] = input_cond[i, 9]
            TR4[i] = input_cond[i, 10]
            u_meas[i,0] = input_cond[i, 2]
            u_meas[i, 1] = input_cond[i, 4]
            u_meas[i, 2] = input_cond[i, 3]

            if input_cond[i, 0] <= input_cond[i, 1]:
                conc_data[i] = (1-conversion_data[i]) * input_cond[i, 0]
            else:
                conc_data[i] = input_cond[i, 0] - (conversion_data[i] * input_cond[i, 1])
            c_std[i] = 0.012
            tr4_std[i] = 1.7
            tr3_std[i] = 0.6
            tr2_std[i] = 0.6
            conv_std[i] = 1.2
            F_123[i,0] = input_cond[i, 0]/conc_st_a*21e-6/t[i]
            F_123[i, 1] = input_cond[i, 0]/conc_st_b*21e-6/t[i]
            F_123[i, 2] = 21e-6/t[i] - F_123[i,0] - F_123[i,1]
            ca_inlet[i] = F_123[i,0]/(F_123[i,0]+F_123[i,2])*conc_st_a
            cb_inlet[i] = conc_st_b
            conv_sd[i] = 1.3

    l0 =9/300# 1/30 #9/300
    la=91/300#1/3 #91/300
    lb=191/300#2/3*9/10 #191/300
    lc=9/300#2/3*1/10#9/300 #2/3*1/10
    n_steps = 10
    L_R = 0.3

    c_inst = []
    T_inst = []
    for j in range(N_p):
         for i in range(N_exp):
            Tc = input_cond[i,2]
            tau = input_cond[i,3]*60 #s
            u = 8.15e-4#input_cond[i,11]#8.15e-4
            Da = 4.7e-5#input_cond[i,12]#4.7e-5
            omega = input_cond[i,4]

            u_meas = [Tc, tau, u, Da, omega,input_cond[i,6]]

            kref = k_p[j,0]/1000 #2.5e-5
            Ea = k_p[j,1]*1000#39200
            lamda_a = (k_cond+rho_m*cpm*Da)/1
            theta1 =[kref,Ea,lamda_a,para_U_z[j,0], para_U_z[j,1], para_U_z[j,2]]
            theta2 = [kref, Ea, lamda_a, para_U_z[j, 3], para_U_z[j, 4], para_U_z[j, 5]]
            theta3 = [kref, Ea, lamda_a, para_U_z[j, 6], para_U_z[j, 7], para_U_z[j, 8]]
            theta4 = [kref, Ea, lamda_a, para_U_z[j, 9], para_U_z[j, 10], para_U_z[j, 11]]

            x_meas[j,i,0,0] = ca_inlet[i]
            x_in_atm = [x_meas[j,i,0,0] * 1000, 0, 0, 0, x_meas[j,i,4,0]]
            ca_adm_nr_atm,T_adm_nr_atm,U_zone_1[j,i] = adm_pfr_nr_atm(x_in_atm, u_meas, theta1, h_scale, L_R*l0, n_steps)
            x_meas[j, i, 0, 1] = ca_adm_nr_atm[-1]
            x_meas[j, i, 4, 1] = T_adm_nr_atm[-1]
            x_in_nr=[x_meas[j, i, 0, 1]*1000, 0, 0,0,x_meas[j, i, 4, 1]]
            ca_adm_nr,T_adm_nr,U_zone_2[j,i] = adm_pfr_nr(x_in_nr, u_meas, theta2, h_scale, L_R*la, n_steps)
            x_meas[j, i, 4, 2] = T_adm_nr[-1]
            x_meas[j, i, 0, 2] = x_meas[j,i,0,0]* ((F_123[i,0]+F_123[i,2])/(F_123[i,0]+F_123[i,1]+F_123[i,2]))
            x_meas[j, i, 1, 2] = conc_st_b * (
                        (F_123[i, 1]) / (F_123[i, 0] + F_123[i, 1] + F_123[i, 2]))
            x_in_na = [x_meas[j, i, 0, 2]*1000, x_meas[j, i, 1, 2]*1000, x_meas[j, i, 2, 1]*1000,x_meas[j, i, 3, 1]*1000,x_meas[j, i, 4, 1]]
            ca_adm_na,T_adm_na,U_zone_3[j,i] = adm_pfr_na(x_in_na, u_meas, theta3, h_scale, L_R*lb, n_steps)
            x_meas[j, i, 0, 3] =ca_adm_na[-1]
            x_meas[j, i, 1, 3] =x_meas[j, i, 1, 2] - (x_meas[j, i, 0, 2]- ca_adm_na[-1])
            x_meas[j, i, 2, 3] = x_meas[j, i, 2, 2]+(x_meas[j, i, 0, 2] - ca_adm_na[-1])
            x_meas[j, i, 3, 3] =  x_meas[j, i, 3, 2]+ (x_meas[j, i, 0, 2] - ca_adm_na[-1])
            x_meas[j, i, 4, 3] = T_adm_na[-1]
            x_in_ad = [x_meas[j, i, 0, 3]*1000, x_meas[j, i, 1, 3]*1000, x_meas[j, i, 2, 3]*1000,x_meas[j, i, 3, 3]*1000,x_meas[j, i, 4, 3]]
            ca_adm_ad,T_adm_ad,U_zone_4[j,i] = adm_pfr_ad(x_in_ad, u_meas, theta4, h_scale, L_R*lc, n_steps)
            x_meas[j, i, 0, 4] =ca_adm_ad[-1]
            x_meas[j, i, 1, 4] =x_meas[j, i, 1, 3] - (x_meas[j, i, 0, 3]- ca_adm_ad[-1])
            x_meas[j, i, 2, 4] = x_meas[j, i, 2, 3]+ (x_meas[j, i, 0, 3] - ca_adm_ad[-1])
            x_meas[j, i, 3, 4] = x_meas[j, i, 3, 3]+ (x_meas[j, i, 0, 3] - ca_adm_ad[-1])
            x_meas[j, i, 4, 4] = T_adm_ad[-1]
            if i==0 and j==0:
                c_inst.append([np.concatenate((ca_adm_nr_atm,ca_adm_nr,ca_adm_na,ca_adm_ad))])
                T_inst.append([np.concatenate((T_adm_nr_atm, T_adm_nr, T_adm_na, T_adm_ad))])

    for j in range(N_p):
        for i in range(N_exp):
            if x_meas[j,i, 0, 2] <= x_meas[j,i, 1,2]:
                conversion_cstr[j,i,0] = (x_meas[j,i, 0, 2] - x_meas[j,i, 0, 3]) / x_meas[j,i, 0, 2] * 100
                conversion_cstr[j, i, 1] = (x_meas[j, i, 0, 2] - x_meas[j, i, 0, 4]) / x_meas[j, i, 0, 2] * 100
                ca_data[i] = input_cond[i, 0]* (1 - conversion_data[i]/100)
                cb_data[i] = input_cond[i, 1] - (input_cond[i, 0]-ca_data[i])
            else:
                conversion_cstr[j,i,0] = (x_meas[j,i, 1, 2] - x_meas[j,i, 1, 3]) / x_meas[j,i, 1, 2] * 100
                conversion_cstr[j, i, 1] = (x_meas[j, i, 1, 2] - x_meas[j, i, 1, 4]) / x_meas[j, i, 1, 2] * 100
                cb_data[i] = input_cond[i, 1]* (1 - conversion_data[i]/100)
                ca_data[i] = input_cond[i, 0] - (input_cond[i, 1]-cb_data[i])

    conversion_cstr[conversion_cstr > 100] = 100


    x_input_gp = np.zeros([N_exp, 5])
    y_output_gp = np.zeros([N_exp, 1])
    for i in range(N_exp):
        x_input_gp[i,:]=input_cond[i, :5]
        y_output_gp[i,:] = ca_data[i] - np.mean(x_meas[:,i, 0, 4],axis=0)

    XX = x_input_gp
    yy = y_output_gp
    all_scores_train = []

    noise_std = 0.024
    os.makedirs("saved_models", exist_ok=True)


    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel,alpha=noise_std**2,

                                   random_state=0, normalize_y=True, n_restarts_optimizer=9)

    model.fit(XX, yy)
    r2_train = model.score(XX, yy)
    all_scores_train.append(r2_train)
    # Save model
    joblib.dump(model, f"saved_models/model1_fold.pkl")

    models = model

    N_ins=100
    prg =N_exp

    x_meas_sample=lhs_exp_cond(N_ins,prg)
    var_t= np.zeros((len(x_meas_sample),3))
    pre_t = np.zeros((len(x_meas_sample)))
    for i in range(len(x_meas_sample)):
        var_t[i,0], pb_m= PB_prediction_var(np.hstack((x_meas_sample[i],input_cond[0,5:])),para_KU, z_sd)
        var_t[i,1], gp_m= GP_var(x_meas_sample[i].reshape(1,-1),models)
        var_t[i,2] = var_t[i,0] + var_t[i,1]
        pre_t[i] = pb_m + gp_m


    x_meas_sample_past = np.zeros([N_exp, 5])
    for i in range(N_exp):
        x_meas_sample_past[i,0] = input_cond[i,0]
        x_meas_sample_past[i, 1] = input_cond[i, 1]
        x_meas_sample_past[i, 2] = input_cond[i, 2]
        x_meas_sample_past[i, 3] = input_cond[i, 3]
        x_meas_sample_past[i, 4] = input_cond[i, 4]

    n_points = 4
    dof = N_exp*5*n_points - z_sd.shape[0]
    conf_level=0.95
    t_test = np.zeros(z_sd.shape[0])
    t_ref = stats.t.ppf((conf_level), dof)
    for i in range(z_sd.shape[0]):
        t_test[i] = para_KU[i] / (z_sd[i, i] * stats.t.ppf((conf_level), dof))

    chi2 = np.zeros((N_exp,3))
    ref_c = stats.chi2.ppf((conf_level), dof)
    hy_pred = np.zeros((N_exp))
    pb_pred = np.zeros((N_exp))
    for i in range(N_exp):
        _, pb_past= PB_prediction_var(input_cond[i],para_KU, z_sd)
        _, gp_past= GP_var(x_meas_sample_past[i].reshape(1,-1),models)
        chi2[i,0] = (((np.mean(x_meas[:,i, 0, 4],axis=0)+gp_past) - ca_data[i])/(c_std[i]))**2
        chi2[i, 1] = ((np.mean(x_meas[:,i,4,4],axis=0) - react_temp_out[i])/tr4_std[i])**2
        chi2[i, 2] = chi2[i,0] + chi2[i,1]
        hy_pred[i] = ((np.mean(x_meas[:,i, 0, 4],axis=0)+gp_past) - ca_data[i])
        pb_pred[i] = pb_past

    #r_squared
    r2 = 1-sum(hy_pred[ki]**2 for ki in range(N_exp))/sum((ca_data[ki]-np.mean(ca_data))**2 for ki in range(N_exp))
    chi2_t = sum(chi2[ki,2] for ki in range(N_exp))

    # best result
    ix = np.argmin(var_t[:,2])
    des_exp=np.hstack((x_meas_sample, var_t[:,2].reshape(-1,1)))
    df1 = pd.DataFrame(des_exp, columns =['a','b','c', 'd', 'e', 'f']).sort_values('f')
    df1 = df1.head(2)
    new_row_1 = [1e9/60*df1.values[0][0]*(21e-6/df1.values[0][3])/conc_st_a,
               1e9/60*df1.values[0][1] * (21e-6 / df1.values[0][3]) / conc_st_b,
               1e9/60*((21e-6 / df1.values[0][3]) - \
               (df1.values[0][0] * (21e-6 / df1.values[0][3]) / conc_st_a) - \
               df1.values[0][1] * (21e-6 / df1.values[0][3]) / conc_st_b),
               df1.values[0][2],
               df1.values[0][4],
               df1.values[0][5],
               chi2_t,ref_c,
               min(abs(t_test)),t_ref,
               r2]

    new_row_2 = [1e9/60*df1.values[1][0]*(21e-6/df1.values[1][3])/conc_st_a,
               1e9/60*df1.values[1][1] * (21e-6 / df1.values[1][3]) / conc_st_b,
               1e9/60*((21e-6 / df1.values[1][3]) - \
               (df1.values[1][0] * (21e-6 / df1.values[1][3]) / conc_st_a) - \
               df1.values[1][1] * (21e-6 / df1.values[1][3]) / conc_st_b),
               df1.values[1][2],
               df1.values[1][4],
               df1.values[1][5],
               chi2_t,ref_c,
               min(abs(t_test)),t_ref,
               r2]

    return new_row_1, new_row_2

start = time.time()
os.chdir('C:/Users/Emmanuel/Documents/DigMan/2025/August 2025/Jayesh')
#os.chdir('C:/Users/Emmanuel/Dropbox/UCL/DigiPROReact')

input_file = 'Imine_Exp_close_loop.csv'
output_file= 'Output Data_close_loop_2.csv'

new_row_1,new_row_2 =close_loop(output_file,input_file)

f = open(input_file, 'a', newline='')
# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(new_row_1)
writer.writerow(new_row_2)

# close the file
f.close()

elapsed_time_fl = (time.time() - start)
print('Elapsed time: ', elapsed_time_fl)