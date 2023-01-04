import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output


stationary_distrib_mm1inf = lambda i,rho_mm1inf : (1-rho_mm1inf)*rho_mm1inf**i
stationary_distrib_mm1K = lambda i,rho_mm1K,K : ((1-rho_mm1K)*rho_mm1K**i) / (1-rho_mm1K**(K+1)) *(i<=K)


def run_mm1inf(lambd: float, mu: float, rng, niter: int):
    """
    Args:
        lambd (float): birth rate.
        mu (float): death rate
        rng (int): random seed (or random generator).
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
    
    T,X=np.zeros(niter),np.zeros(niter)
    for i in range(1,niter):
        if X[i]==0:
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


def plotHistogramm(X,weights,rho,alpha=0.2):
    valeurs_prises = np.unique(X)
    plt.hist(X,bins=valeurs_prises,density=True,weights=weights)
    plt.bar(valeurs_prises,stationary_distrib_mm1inf(valeurs_prises,rho),color='r',alpha=0.3)


def theorical_mean(rho_mm1K,K):
        return np.sum([i*stationary_distrib_mm1K(i,rho_mm1K,K) for i in range(K+1)])



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
   