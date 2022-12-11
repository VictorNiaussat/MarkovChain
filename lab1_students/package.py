import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns

def tirer_dans_mu(mu):
    H = np.argwhere(np.cumsum(mu)<np.random.default_rng().random())
    if len(H)==0:
        return (0)
    return(np.max(H))

def ehrenfest(mu=15,Generator = np.random.default_rng(),nmax = 1000):
    state = mu
    Liste_Ehrenfest=[mu]

    for j in range(nmax):
        if Generator.random() < state/K:
            state+=-1
        else :
            state+=1
        Liste_Ehrenfest.append(state)
    return(Liste_Ehrenfest)

def exercice1():
    K = 30
    p=1/2
    diagonal = np.array([i/K for i in range(1,K+1)])
    P = np.diag(diagonal,k=-1)+np.diag(1+1/K-diagonal,k=1)
    B = binom(K, p)
    x = range(K+1)
    mu_bin = B.pmf(x)
    plt.bar(x,mu_bin)
    print(mu_bin.T@P - mu_bin)
    print(f"On vérifie ainsi que µ est une mesure invariante")
    print(f"***************************************************************************")
    print(f"*******************          EHRENFEST          ***************************")
    print(f"***************************************************************************")

    simulate_ehrenfest = ehrenfest(mu=0,Generator = np.random.default_rng(),nmax = 5000)
    plt.plot(simulate_ehrenfest)
    hist = plt.hist(simulate_ehrenfest,density=True,bins=K-5)
    plt.hist(simulate_ehrenfest,density=True,bins=K-5,color="red",label="Histogramme de densité des valeurs simulées")
    plt.bar(x,mu_bin,color = "grey",label="Histogramme loi binomiale")
    plt.legend()
    plt.show()
