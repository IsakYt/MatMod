from RHS import fatCircle
from RHS import fatLap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root

def solver(N,r0, n0, steps, DX, dt, fat = True):
    """
    input:
        N: int, number of grid-points across circle diameter
        r0: array, initial distribuiton of receptors
        n0: array, initial distribution of neurotransmitters
        steps: int, number of time evaluations
        DX: float, length of spacial step size
        dt: float, time step length
    return:
        r,n
    """

    #Constants needed for scaling
    N0 = 3.64 
    R0 = 0.11
    L = 0.22*10**(-6)
    Tscale = 10**(-10)
    alpha = K/(L**2) * Tscale
    M = len(r0)

    #Arrays for storing the temporal evolutions
    r = np.zeros((steps,M),dtype = float)
    n = np.copy(r)
    
    #Setting initial conds
    r[0] = r0
    n[0] = n0
        
    
    #The sparse matrix representing the laplacian operator, named A in the report.
    lap = fatLap(N)
    lap = alpha*lap #alpha is the heat diffusion constant with scaling
    fired = False

    #f and g are the functions which handle the fixed point iteration (FPI).
    def f(V,n_1,r_1):
            #V[0] = n[i+1]
            #V[1] = r[i+1]
        return r_1 - V[1] - dt*N0*kon*Tscale*np.multiply(V[0],V[1]) + dt*Tscale*koff*(1-V[1])

    def g(V,n_1,r_1):
        return n_1 - V[0] + dt*lap.dot(V[0]) - dt*(R0*kon*Tscale)*np.multiply(V[0],V[1]) + dt*(R0*koff*Tscale/N0)*(1-V[1])

    #temporal evolution done here
    for i in range(steps-1):    
        #implicit approach
        guess = [n[i],r[i]] #initial guess for FPI
        f1 = lambda P: [g(P,n[i],r[i]), f(P,n[i],r[i])] #FPI needs a vector function
        vec = root(f1,guess, method = "krylov", tol = 1e9).x #FPI done with scipy.optimize.root, krylov approximates the jacobian.

        #update the storage arrays
        n[i+1] = vec[0]
        r[i+1] = vec[1]

        #check if signal has fired
        if not fired and sum(r[i+1])/sum(r0) < 0.5:
            print(f"Transmission after {i*dt}, row {i}")
            fired = True

    return r,n

#comment
def Initial(P, x = 1, y = 0, r = 0.1):
    """
    Makes a disc of value 1

    """
    arr = np.where((P[:,0]-0.5)**2 + (P[:,1]-0.5)**2 < r**2, x, y)
    return arr


if __name__ == "__main__":
    N = 30
    DX = 1/N
    DT = 0.01

    pulse = 5e6

    #Initial conds for r and n are handled here by calling Initial()
    r0 = fatCircle(N)
    r0 = Initial(r0, r = 0.2)
    r0 = r0

    n0 = fatCircle(N)
    n0 = Initial(n0, r = 0.1)
    n0 = pulse*n0 
    print(np.sum(n0), "here")

    #constants which are used.
    kon = 4e3
    koff = 5
    K = 8e-7
    Times = 200

    rec, nevro = solver(N, r0, n0, Times, DX, DT, fat = True)

    points = fatCircle(N)
    print(sum(nevro[-1]/sum(nevro[0])),"there")
    print(sum(rec[-1])/sum(rec[0]))

    #Below is plotting and animation.
    fig = plt.figure(figsize = (4,4))
    ax1 = fig.add_subplot(121,projection = "3d")
    ax2 = fig.add_subplot(122,projection = "3d")
    ax1.plot_trisurf(points[:,0], points[:,1], nevro[0], cmap='inferno')
    ax2.plot_trisurf(points[:,0], points[:,1], nevro[89], cmap='inferno')
    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("y-axis")
    ax2.set_xlabel("x-axis")
    ax2.set_ylabel("y-axis")
    plt.show()
    plt.close()


    fig = plt.figure()
    ax1 = fig.add_subplot(121,projection = "3d")
    ax2 = fig.add_subplot(122,projection = "3d")
    ax1.plot_trisurf(points[:,0], points[:,1], rec[0], cmap='inferno')
    ax2.plot_trisurf(points[:,0], points[:,1], rec[89], cmap='inferno')
    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("y-axis")
    ax2.set_xlabel("x-axis")
    ax2.set_ylabel("y-axis")
    plt.show()
    plt.close()
    #Here comes code for animation:
    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Create a figure and a set of subplots
    fig = plt.figure()
    ax = plt.subplot(projection = '3d')

    # Method to change the contour data points
    def animate(i):
        ax.clear()
        #ax.set_zlim(-1,1)
        ax.set_title(f"Time elapsed {i * DT:.8f}")
        ax.plot_trisurf(points[:,0], points[:,1], nevro[i], cmap='inferno') 

    # Call animate method
    ani = animation.FuncAnimation(fig, animate, Times, interval = 10, blit=False)

    # Display the plot
    plt.show()