import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import plotly.express as px
import package as pack

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

    simulate_ehrenfest = pack.ehrenfest(mu=0,Generator = np.random.default_rng(),nmax = 5000)
    plt.plot(simulate_ehrenfest)
    hist = plt.hist(simulate_ehrenfest,density=True,bins=K-5)
    plt.hist(simulate_ehrenfest,density=True,bins=K-5,color="red",label="Histogramme de densité des valeurs simulées")
    plt.bar(x,mu_bin,color = "grey",label="Histogramme loi binomiale")
    plt.legend()
    plt.show()
    print(f"Le temps de retour pour un K=10 est de (simulation sur un trajet): {pack.return_time_0(K=10,Generator = np.random.default_rng())}")
    print(f"***************************************************************************")
    print(f"*******************      Run several chains    ****************************")
    print(f"***************************************************************************")

    T_list = [pack.return_time_0(K=10) for i in range(2000)]
    plt.hist(T_list,bins=100,density=True)
    plt.show()
    L_k=[]
    for K in range(1,20):
        T_list = [pack.return_time_0(K) for i in range(2000)]
        L_k.append(np.mean(T_list))

    plt.plot(range(1,20),L_k)



def exercice2():
    P = np.array([
        [0.2,0.7,0.1],
        [0.9,0,0.1],
        [0.2,0.8,0]])
    distribution_init = [0,1,0]
    nMax=2000
    states = pack.simulate_dthmc(transitionMatrix=P,distribution=distribution_init,nMax=nMax)
    states=states.astype(int)
    fig = px.histogram(states,histnorm='probability',title='Histogramme des états visités',nbins=3)
    fig.update_traces(xbins_size=0.1)
    fig.update_layout(
        xaxis_title="Etats de la chaine",
        yaxis_title="Probabilité de visite")
    fig.show()
    w,v=np.linalg.eig(P.T)
    normalized_vector = np.real(v[:,0] / np.sum(v[:,0]))
    print(f"Valeurs propres :                                       {w}")
    print(f"Vecteur propre associé à la valeur propre 1 :           {v[:,0]}")
    print(f"Vecteur propre associé à la valeur propre 1 normalisé : {normalized_vector}")



    fig = px.histogram(normalized_vector,title='Histogramme du vecteur propre associé à la valeur propre 1',nbins=3,labels={"0":"Etat 0","1":"Etat 1","2":"Etat 2"})
    fig.update_traces(xbins_size=0.01)
    fig.update_layout(
        xaxis_title="Etats de la chaine",
        yaxis_title="Probabilité de visite")
    fig.show()
    nStep=50
    distribution_n,history = pack.compute_distrib_n(P,distribution_init,nstep=nStep)
    plt.hist(distribution_n,label=['etat 0','etat 1','etat 2'])
    plt.legend()
    print(f"La distribution au step {nStep} vaut : {distribution_n}")
    history = np.array(history)
    history= history.reshape(50,3)
    plt.figure()
    plt.plot(history[:,0],label="Evolution de la distribution en 0")
    plt.plot(history[:,1],label = "Evolution de la distribution en 1")
    plt.plot(history[:,2],label = "Evolution de la distribution en 2")
    plt.xlabel("Nombre de steps")
    plt.legend()
    pi_vect = np.array([normalized_vector for i in range(0,nStep)])
    diff = np.linalg.norm(history - pi_vect,ord=1,axis=1)
    plt.figure()
    plt.plot(diff)
    plt.title("Evolution de l'écart entre la mesure invariante et son estimation")
    plt.xlabel("Nombre de steps")
    plt.ylabel("Norme l1 de la différence")
    plt.ylim(0,1)
    plt.show()
    MoyennePerState=[]
    for state_considered in range(0,3):
        states = pack.simulate_dthmc(transitionMatrix=P,distribution=distribution_init,nMax=nMax)
        positionOfReturn = []
        for step  in range(len(states)):
            if states[step]==state_considered:
                positionOfReturn.append(step)
        MoyennePerState.append(pack.computeMoyenneReturn(positionOfReturn))
    print(f"Le nombre moyen de retour pour chaque état est défini par (moyenne empirique)")
    print(f"Moyenne du nombre de step pour le retour en 0 : {round(MoyennePerState[0],2)}")
    print(f"Moyenne du nombre de step pour le retour en 1 : {round(MoyennePerState[1],2)}")
    print(f"Moyenne du nombre de step pour le retour en 2 : {round(MoyennePerState[2],2)}")
    tabTempsRetour = [1/val for val in normalized_vector]
    print(f"Moyenne du temps de retour en 0 : {round(tabTempsRetour[0],2)}")
    print(f"Moyenne du temps de retour en 1 : {round(tabTempsRetour[1],2)}")
    print(f"Moyenne du temps de retour en 2 : {round(tabTempsRetour[2],2)}")




