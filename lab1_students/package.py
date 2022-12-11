import numpy as np


P = np.array([
    [0.2,0.7,0.1],
    [0.9,0,0.1],
    [0.2,0.8,0]])
distribution_init = [0,1,0]

def tirer_dans_mu(mu):
    H = np.argwhere(np.cumsum(mu)<np.random.default_rng().random())
    if len(H)==0:
        return (0)
    return(np.max(H))

def ehrenfest(mu=15,K=30,Generator = np.random.default_rng(),nmax = 1000):
    state = mu
    Liste_Ehrenfest=[mu]

    for j in range(nmax):
        if Generator.random() < state/K:
            state+=-1
        else :
            state+=1
        Liste_Ehrenfest.append(state)
    return(Liste_Ehrenfest)

def return_time_0(K=10,Generator = np.random.default_rng(),nmax=5000):
    state = 1
    T=1
    while state!=0 and T<=nmax:
        if Generator.random() < state/K:
            state+=-1
        else :
            state+=1
        T+=1
    return(T)



def simulate_dthmc(transitionMatrix,distribution,nMax=100,generator = np.random.default_rng(100)):
    generator = generator
    labelStates=np.array([i for i in range(0,len(distribution))])
    states = np.array([])
    states = np.append(states,int(np.argwhere(np.array(distribution)==1)[0]))   # initial State
    for step in range(1,nMax):
        next_state = np.random.choice(labelStates,p=P[int(states[step-1]),:])
        states=np.append(states,next_state)
    return states

def compute_distrib_n(TransitionMatrix=P, distribution=None, nstep=100):
    if distribution is None:
        distribution = distribution_init
    distribution_n = np.array([distribution_init])
    historyDistrib = []
    for step in range (0,nstep):
        distribution_n=np.dot(distribution_n,P)
        historyDistrib.append(distribution_n)
    return distribution_n,historyDistrib

def computeMoyenneReturn(positionOfReturn):
    n=len(positionOfReturn)
    tabNbStepReturn = []
    for i in range(1,n):
        tabNbStepReturn.append(positionOfReturn[i]-positionOfReturn[i-1])
    return np.mean(tabNbStepReturn)


