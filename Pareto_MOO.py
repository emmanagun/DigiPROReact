import pandas as pd
import os
import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.integrate import ode
from math import exp, log, pi
import matplotlib.pyplot as plt
import time
import math
import joblib
from scipy import stats
from pyDOE2 import lhs, fullfact
import sympy as sp
from numpy.polynomial.legendre import Legendre

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn import preprocessing
import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
from sklearn.utils import shuffle
import seaborn as sn
from numpy.random import seed
from scipy import r_
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import pickle
from scipy.stats import qmc

def lhs_exp_cond(N_exp,prng):
    # Specification of experimental condition
    if N_exp ==1:
        l_bounds = [0.100,0.500, 5, 1., 100]
        u_bounds = [5.00, 2.00, 25, 10., 300]
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
        control_bounds = [0.100,4.00, 0.500, 2.00, 5, 25, 1., 10., 100, 300]
        c_b = np.array(control_bounds).reshape(5, 2)

        #n_phi = c_b.shape[0]  #
        n_phi = 5
        # This code generates DoE by Latin hypercube sampling
        #prng = 3
        lhs_points = lhs(n_phi, samples=N_exp, criterion='maximin', iterations=50, random_state=prng)
        lhs_actual = np.zeros([N_exp, n_phi])
        for i in range(N_exp):
            lhs_actual[i, 0] = lhs_points[i, 0] * (c_b[0, 1] - c_b[0, 0]) + c_b[0, 0]
            lhs_actual[i, 1] = (lhs_points[i, 1] * (c_b[1, 1] - c_b[1, 0]) + c_b[1, 0])* lhs_actual[i, 0]
            lhs_actual[i, 2] = lhs_points[i, 2] * (c_b[2, 1] - c_b[2, 0]) + c_b[2, 0]
            lhs_actual[i, 3] = lhs_points[i, 3] * (c_b[3, 1] - c_b[3, 0]) + c_b[3, 0]
            lhs_actual[i, 4] = round(lhs_points[i, 4] * (c_b[4, 1] - c_b[4, 0]) + c_b[4, 0])
    return lhs_actual

start = time.time()

os.chdir('C:/Users/Emmanuel/Documents/DigMan/2026/June 2026')

data=pd.read_excel('Data for Pareto.xlsx')
input_cond=data[['Benzaldehyde inlet conc (M)','Benzylamine inlet conc (M)','Reactor Temp. ( C)','space time (mins)'
    ,'TVR speed (rev./mins)','conv','power','c_err']].values
print(len(data))
print(len(input_cond))
N_exp = len(input_cond)

x_input_gp = np.zeros([N_exp, 5])
y_output_gp = np.zeros([N_exp, 2])
for i in range(N_exp):
    x_input_gp[i,:]=input_cond[i, :5]
    y_output_gp[i,:] = input_cond[i, 5:7]

X = x_input_gp
y = y_output_gp
X,y = shuffle(X,y)

all_scores_train = []
sd_train = []
all_scores = []
sd = []

all_scores_train_2 = []
sd_train_2 = []
all_scores_2 = []
sd_2 = []

noise_std = 0.024

kernel1 = DotProduct() + WhiteKernel()
kernel2 = 1.0 * RBF(1.0)
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
model1 = GaussianProcessRegressor(kernel=kernel,

                                  random_state=0, normalize_y=True, n_restarts_optimizer=9)

model1.fit(X, y[:, 0])
r2 = model1.score(X, y[:,0])
print(r2)

# Save the trained model as a pickle string.
saved_model_1 = pickle.dumps(model1)

# Load the pickled model
model_1_from_pickle = pickle.loads(saved_model_1)

# Use the loaded pickled model to make predictions
print(model_1_from_pickle.predict(X))

# Save the model as a pickle in a file
joblib.dump(model1, 'gp_model_1.pkl')

# Load the model from the file
model_1_from_joblib = joblib.load('gp_model_1.pkl')

model2 = GaussianProcessRegressor(kernel=kernel,

                                  random_state=0, normalize_y=True, n_restarts_optimizer=9)

model2.fit(X, y[:, 1])
y1_model = model1.predict(X)
#y1_model_std=model1.predict(X, return_std=False, return_cov=False)[]
y2_model = model2.predict(X)
r2_2 = model2.score(X, y[:,1])
print(r2_2)

# Save the trained model as a pickle string.
saved_model_2 = pickle.dumps(model2)

# Load the pickled model
model_2_from_pickle = pickle.loads(saved_model_2)

# Use the loaded pickled model to make predictions
print(model_2_from_pickle.predict(X))

# Save the model as a pickle in a file
joblib.dump(model2, 'gp_model_2.pkl')

# Load the model from the file
model_2_from_joblib = joblib.load('gp_model_2.pkl')


# Adaptive weights exploration
N_ins=100
prg =4
x_values = lhs_exp_cond(N_ins,prg)
pareto_solutions = []
best_des_x = []
f1=model1
f2=model2
# We'll loop over weights that change during optimisation
f1_vals = []
f2_vals = []
for step in range(0, 21):
    w1 = step / 20          # Weight for f1
    w2 = 1 - w1             # Weight for f2

    # Weighted sum for this step
    scores = []
    for dd in range(len(x_values)):
        scores.append(w1*(f1.predict(x_values[dd,:].reshape(1,-1))[0]-max(y_output_gp[:,0]))/max(y_output_gp[:,0]) -
                      w2*(f2.predict(x_values[dd,:].reshape(1,-1))[0]-min(y_output_gp[:,1]))/min(y_output_gp[:,1]))
    #scores = [w1*f1.predict(x_values[dd,:].reshape(1,-1))[0] + w2*f2.predict(x_values[dd,:].reshape(1,-1))[0] for dd in range(len(x_values))]
    best_index = np.argmin(scores)
    best_x = x_values[best_index]
    best_des_x.append(best_x)
    pareto_solutions.append((f1.predict(best_x.reshape(1,-1))[0], -f2.predict(best_x.reshape(1,-1)))[0])
    f1_vals.append(f1.predict(best_x.reshape(1,-1))[0])
    f2_vals.append(f2.predict(best_x.reshape(1, -1))[0])


coefficients=np.polyfit(np.array(f2_vals),np.array(f2_vals),2)
poly_model = np.poly1d(coefficients)

plt.figure(figsize=(6, 5))
plt.plot(f1_vals, f2_vals, 'o-', label="Adaptive weights solutions")
#plt.plot(list(y_output_gp[:,0]), list(-y_output_gp[:,1]), 'p-', label="data")
#plt.scatter(list(y_output_gp[:,0]), list(-y_output_gp[:,1]), 'p-', label="data")
plt.xlabel("Objective 1 (conversion)", fontsize=35)
plt.ylabel("Objective 2 (power)", fontsize=35)
plt.title("Pareto front exploration with adaptive weights")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend()
'''
plt.fill_between(
        x= np.array(f1_vals),
        y1= poly_model(np.array(f1_vals)),
        where= (30 < np.array(f1_vals))&(np.array(f1_vals) < 90),
        color= "b",
        alpha= 0.2)

'''
plt.show()

plt.plot(y[:,0],y[:,0], color='black')
plt.scatter(y[:,0],y1_model, color='red', marker="X",s=100)
plt.errorbar(y[:,0],y[:,0],yerr=(2*input_cond[:,7]),color='black',capsize=10, linestyle='none',marker='o')
plt.errorbar(y[:,0],y1_model,yerr=(2*model1.predict(X, return_std=True, return_cov=False)[1]),color='red',capsize=10, linestyle='none',marker='o')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Data x1 (conversion)', fontsize=35)
plt.ylabel('GP1 prediction', fontsize=35)
plt.legend(['Data','GP1'], fontsize=35)
labels =list(range(1,len(y[:,0])+1))
for label, x, yy in zip(labels, y[:,0], y1_model):
    plt.annotate(label, xy=(x,yy), xytext=(-30, 30), textcoords='offset points', ha='right', va='bottom',size=25,
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.95), arrowprops=dict(arrowstyle='->',
                           connectionstyle='arc3,rad=0'))

plt.show()


plt.plot(y[:,1],y[:,1], color='black')
plt.scatter(y[:,1],y2_model, color='red', marker="X",s=100)
#plt.errorbar(y_output_gp[:,1],y_output_gp[:,1],yerr=(2*input_cond[:,7]),color='black',capsize=10, linestyle='none',marker='o')
plt.errorbar(y[:,1],y2_model,yerr=(2*model2.predict(X, return_std=True, return_cov=False)[1]),color='red',capsize=10, linestyle='none',marker='o')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Data x2 (power) (W)', fontsize=35)
plt.ylabel('GP2 prediction (W)', fontsize=35)
plt.legend(['Data','GP2'], fontsize=35)
labels =list(range(1,len(y[:,1])+1))
for label, x, y in zip(labels, y[:,1], y2_model):
    plt.annotate(label, xy=(x,y), xytext=(-30, 30), textcoords='offset points', ha='right', va='bottom',size=25,
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.95), arrowprops=dict(arrowstyle='->',
                           connectionstyle='arc3,rad=0'))

plt.show()
