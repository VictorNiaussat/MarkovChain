import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output



#***********************************************************************************************************************
#                                               EXERCICE 1
#***********************************************************************************************************************

stationary_distrib_mm1inf = lambda i,rho_mm1inf : (1-rho_mm1inf)*rho_mm1inf**i
stationary_distrib_mm1K = lambda i,rho_mm1K,K : ((1-rho_mm1K)*rho_mm1K**i) / (1-rho_mm1K**(K+1)) *(i<=K)


def run_mm1inf(lambd: float, mu: float, rng, niter: int):
    """
    Args:
        lambd (float): birth rate.
        mu (float): death rate
        rng : random seed (or random generator).
        niter (int): number of changes (events) in 
                     the process.
    Raises:
        ValueError: error triggered if lambd <= 0.
        ValueError: error triggered if mu <= 0.
    Returns:
        X (array_like): trajectory (X(t_n)).
        T (array_like): time instants at which a change in 
                        the process occurs (t_n).
    """ 
    assert lambd>0, print("lambd<=0")
    assert mu >0, print("mu<=0")
    lambd=1/lambd
    mu=1/mu
    T,X=np.zeros(niter),np.zeros(niter)

    for i in range(1,niter):
        if X[i-1]==0:
            T[i]=T[i-1]+rng.exponential(lambd)
            X[i]=X[i-1]+1
        else:
            Tb = rng.exponential(lambd)
            Td = rng.exponential(mu)
            if Tb<Td : 
                T[i]=T[i-1]+Tb
                X[i]=X[i-1]+1
            else:
                T[i]=T[i-1]+Td
                X[i]=X[i-1]-1
    return X,T

def run_mm1K(lambd: float, mu: float,K : int, rng, niter: int):
    """
    Args:
        lambd (float): birth rate.
        mu (float): death rate
        rng : random seed (or random generator).
        niter (int): number of changes (events) in
                     the process.
    Raises:
        ValueError: error triggered if lambd <= 0.
        ValueError: error triggered if mu <= 0.
    Returns:
        X (array_like): trajectory (X(t_n)).
        T (array_like): time instants at which a change in
                        the process occurs (t_n).
    """
    assert lambd>0, print("lambd<=0")
    assert mu >0, print("mu<=0")
    lambd=1/lambd
    mu=1/mu

    T,X=np.zeros(niter),np.zeros(niter)

    for i in range(1,niter):
        if X[i-1] == 0:
            T[i]=(T[i-1]+rng.exponential(lambd))
            X[i]=(X[i-1]+1)
        elif X[i-1] == K:
            T[i]=(T[i-1]+rng.exponential(mu))
            X[i]=(X[i-1]-1)
        else:
            Tb = rng.exponential(lambd)
            Td = rng.exponential(mu)
            if Tb<Td :
                T[i]=(T[i-1]+Tb)
                X[i]=(X[i-1]+1)
            else:
                T[i]=(T[i-1]+Td)
                X[i]=(X[i-1]-1)
    return X,T


def draw_queue(X,T):
    """
    :param X (array-like): évolution du nombre de personnes dans la file
    :param T (array-like): temps associés aux changements du nombre de personnes dans la file
    :return: None, affiche un graphique pour l'évolution
    """
    plt.style.use('ggplot')
    plt.step(T,X)
    plt.title("Evolution du nombre de personnes dans la file en fonction du temps")
    plt.xlabel("Temps")
    plt.ylabel("Nombre de personnes dans la file")
    plt.show()


def plotHistogramm(X,weights,rho,alpha=0.2,step_numbers = 1000,affiche_step=False):
    plt.style.use('ggplot')
    valeurs_prises = np.unique(X)
    plt.hist(X,bins=valeurs_prises,density=True,weights=weights,label='Valeurs simulées',color='b')
    plt.bar(valeurs_prises,stationary_distrib_mm1inf(valeurs_prises,rho),color='r',alpha=alpha,label='Valeurs théoriques')
    st="Histrogramme de la proportion de personnes dans la queue"
    if affiche_step:
        st=st+f' - nombre de steps : {step_numbers}'
    plt.title(st)
    plt.xlabel("Nombre de personnes")
    plt.ylabel('Proportion')
    plt.legend()
    plt.show()

def plotHistogrammK(X,weights,rho,K,alpha=0.2,step_numbers = 1000,affiche_step=False):
    plt.style.use('ggplot')
    valeurs_prises_mm1K= np.unique(X)
    a,bins, _= plt.hist(X[:-1],bins=range(K+2),density=True,weights=np.diff(weights),label='Valeurs simulées',color='b')
    plt.bar(valeurs_prises_mm1K+0.5,stationary_distrib_mm1K(valeurs_prises_mm1K,rho,K),alpha=alpha,color='r',label='Valeurs théoriques')
    st = "Histrogramme de la proportion de personnes dans la queue"
    if affiche_step:
        st = st+f' - nombre de steps : {step_numbers}'
    plt.title(st)
    plt.xlabel("Nombre de personnes")
    plt.ylabel('Proportion')
    plt.legend()
    plt.show()

def theorical_mean(rho_mm1K,K):
        return np.sum([i*stationary_distrib_mm1K(i,rho_mm1K,K) for i in range(K+1)])



#***********************************************************************************************************************
#                                               EXERCICE 2
#***********************************************************************************************************************

def metro_hasting_ising_2D(N,beta,num_samples,display_iter):
    """
    Args:
        N (int) : grid dimension grille (N x N)
        beta (float) : inverse temperature 
        num_samples (int) : number of iterations on the grid
        display_iter (int) : number of steps before refreshing grid display
    Raises:
        ValueError: error triggered if display_iter <= 0.
        ValueError: error triggered if num_samples <= 0.
        ValueError: error triggered if N <= 0.
    
    """
    assert N>0, print(f"N should be positive, N={N}")
    assert display_iter>0, print(f"display_iter should be positive, display_iter={display_iter}")
    assert num_samples>0, print(f"num_samples should be positive, num_samples={num_samples}")

    grid = 2*np.random.randint(2, size=(N,N))-1
    for i in range(num_samples):
        x = np.random.randint(N)
        y = np.random.randint(N)
        delta_E = 2 * grid[x,y] * (grid[(x+1)%N,y] + grid[(x-1)%N,y] + grid[x,(y+1)%N] + grid[x,(y-1)%N])
        if np.random.rand() < np.exp(-beta * delta_E):
            grid[x,y] = -grid[x,y]

        if i % display_iter == 0:
            clear_output(wait=True)
            plt.imshow(grid, cmap="gray", vmin=-1, vmax=1)
            plt.show()



#***********************************************************************************************************************
#                                               EXERCICE 3
#***********************************************************************************************************************


def draw_neighbour(xn):
    """
    :param xn(float):paramètre considéré
    :return: un voisin du paramètre à +- 0.005 près selon une loi uniforme
    """
    return(xn+0.005 if np.random.uniform()<0.5 else xn-0.005 )
def acceptance_probability(f,y,xn,Tn):
    """
    :param f(lambda): fonction considérée
    :param y: voisin de xn
    :param xn: particule considérée
    :param Tn: Température pour la probabilité d'acceptation
    :return: (float) dans [0,1], probabilité d'acceptation
    """
    return(1 if f(y)<f(xn) else np.exp(-(f(y)-f(xn))/Tn))

def simulated_annealing(f,N,T_min,x0,T0,draw_neighbour):
    """
    :param f(lambda): fonction considérée
    :param N: nombre d'itérations avant la fin de l'algorithme
    :param T_min: précision maximale
    :param x0: particule de départ
    :param T0: température initiale pour la probabilité d'acceptance
    :param draw_neighbour: Fonction qui génère un voisin 
    :return: tuple (X,F) contenant deux array respectivempent : évolution de x, évolution de f(x)
    """
    n = 0
    xn = x0
    Tn=T0
    X,F=[],[]

    while n<N and Tn>T_min:
        y = draw_neighbour(xn)
        u = np.random.uniform()
        xn = y if u <= acceptance_probability(f,y,xn,Tn) else xn
        X.append(xn)
        F.append(f(xn))
        Tn = T0/np.log(n+2)
        n=n+1
    return(X,F)

def plotAnnealing(X,step_number=1000,drawSteps = False):
    """
    :param X (array-like): évolution du paramètre x
    :param step_number (int): nombre de steps considérés
    :param drawSteps (Boolean): affichage ou non du nombre d'étapes
    :return: None, affiche le graphique de l'avolution de X
    """
    plt.style.use('ggplot')
    plt.plot(X)
    st="Convergence du minimum x recherché"
    if drawSteps:
        st=st+f" - Nombre d'itérations : {step_number}"
    plt.title(st)
    plt.xlabel("itérations")
    plt.ylabel("Valeur de x")
    plt.show()



def create_city(K,max_size_city,etalement):
    """
    :param K(int): nombre de villes à tirer
    :param max_size_city(int): valeur maximale des coordonnées
    :param etalement (int): variance considérée
    :return: (array-like) une grille contenant les coordonnées des villes sur un plan 2D
    """
    grid=[]
    while len(grid)<K:
        grid.append(np.random.randint(max_size_city, size=2)+np.random.normal(scale = etalement/10,size = 2))
    grid = np.array(grid)
    return grid

import pdb
def distance_permut(sigma,dist,grid):
    """
    :param sigma(array-like): permutation considérée
    :param dist(lambda): calcule une norme entre deux vecteurs (ici distance)
    :param grid(array-like): grille considérée des villes par leur coordonnées
    :return: (int) coût total de cette permutation
    """
    K=len(sigma)
    return np.sum([dist(grid[sigma[j]], grid[sigma[(j + 1) % K]]) for j in range(len(grid))])

def generate_indices(K):
    """
    :param K(int): indice maximal à considérer
    :return: tuple(int,int) deux indices différents parmis [0,K]
    """
    i = np.random.randint(0, K)
    k = i
    while k == i:
        k = np.random.randint(0, K)
    return min(i,k), max(i,k)

def draw_neighbourTSP(sigman):
    """
    :param sigman(array-like): tableau des permutations condidérées
    :return: (array-like) permutation aléatoire de deux éléments différents
    """
    sigmanp1 = np.copy(sigman)
    i,k = generate_indices(len(sigman))
    sigmanp1[i:k+1] = np.flip(sigmanp1[i:k+1])
    return(sigmanp1)


def plotTSP(grid):
    """
    :param grid(array-like): grille considérée des villes par leur coordonnées
    :return: None,affiche le graphe de l'évolution
    """
    plt.style.use('ggplot')
    plt.grid()
    plt.scatter(grid[:,0],grid[:,1],marker = "P",color = "red")
    plt.title("Villes du problème considéré")
    plt.xlabel("coordonée x")
    plt.ylabel("coordonnée y")
    plt.show()

def plot_distance_iter(F,N):
    """
    :param F(array-like): Liste des distances au fur et à mesure des itérations
    :param N(int): nombre d'itérations
    :return: None,affiche le graphe de l'évolution
    """
    plt.plot(F,color = "black",linewidth = 2)
    plt.title(f"Distance du chemin en fonction \n du nombre d'itérations ($N = {N} $)")
    plt.xlabel("Nb itérations")
    plt.ylabel("Distance")
    plt.show()
    
def plotTSP_path(paths, points, num_iters=1):

    """
    :path: chemin 
    :points: coordonées des points
    :num_iters: nombre d'itérations
    :return: None,affiche le graphe de l'évolution
    """

    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    
    plt.plot(x, y, 'co')


    a_scale = float(max(x))/float(100)


    if num_iters > 1:

        for i in range(1, num_iters):

 
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]), 
                    head_width = a_scale, color = 'r', 
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))


    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
            color ='red', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'red', length_includes_head = True)

 
    plt.xlim(min(x)-0.1*max(x), max(x)*1.1)
    plt.ylim(min(y)-0.1*max(y), max(y)*1.1)
    plt.show()
    
    
def evolution_travelsaleman(paths,grid,iter_display):
    """
    :paths(array): Tous les chemin dans un array 
    :param grid(array-like): grille considérée des villes par leur coordonnées
    :iter_display(int): Itération d'affichage
    :return: None,affiche le graphe de l'évolution
    """
            
    for _,p in enumerate(paths):
        if _%iter_display == 0:
            clear_output(wait=True)
            plotTSP_path([p], grid, num_iters=1)
