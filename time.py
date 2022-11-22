from RHS import build
from RHS import fatCircle
from RHS import fatLap
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root
from scipy.spatial import Delaunay
#from mpl_toolkits.mplot3d import Axes3D

def solver(N,r0, n0, steps, DX, dt, fat = False):
    """
    input:
        r0: array, initial distribuiton of receptors
        n0: array, initial distribution of neurotransmitters
        q0: array, initial distribution of bound transmitters
        All of the above should have the same shape M*N
        steps: int, number of time evaluations
        DX: arr, the values for dx and dy
        dt: float, time step length
    return:

    """
    if fat:
        N0 = 3.64 
        R0 = 0.11
        M = len(r0)
        r = np.zeros((steps,M),dtype = float)
        n = np.copy(r)

        r[0] = r0
        n[0] = n0

        L = 0.22*10**(-6)
        Tscale = 10**(-9)
        alpha = K/(L**2) * Tscale

        lap = fatLap(N)
        lap = alpha*lap/DX[0]**2

        def f(V,n_1,r_1):
            #V[0] = n[i+1]
            #V[1] = r[i+1]
            return r_1 - V[1] - dt*N0*kon*Tscale*V[0]*V[1] + dt*Tscale*koff*(1-V[1])

        def g(V,n_1,r_1):
            return n_1 - V[0] + dt*lap.dot(V[0]) - dt*(R0*kon*Tscale)*V[0]*V[1] + dt*(R0*koff*Tscale/N0)*(1-V[1])

        for i in range(steps-1):
            #implicit approach

            f1 = lambda P: [f(P,n[i],r[i]), g(P,n[i],r[i])] 
            vec = root(f1,[n[i],r[i]], method = "krylov", tol = 1e-6)
            vec = vec.x
        
            n[i+1] = vec[0]
            r[i+1] = vec[1]


        return r,n

    else:
        M, N = np.shape(r0)
        N0 = 3.64 
        R0 = 0.11 
        r = np.zeros((steps,M*N),dtype = float)
        n = np.copy(r)

        r[0] = r0.flatten()
        n[0] = n0.flatten()
        
        bands, boundary = build(M,N)
        totBound = boundary[0] + boundary[1] + boundary[2] + boundary[3]

        #Use boundary vector to remove boundary indexes 
        #from n,r,q-vector?
        dx,dy = DX[0], DX[1]
        L = 0.44*10**(-6)
        Tscale = 10**(-3)
        alpha = K/(L**2) * Tscale
        # Lx = 0.44*10**(-6)
        # Ly = 0.44*10**(-6)
        # koeff1 = (K/(kon*N0)) * ((1/Lx*dx)**2 + (1/(Ly*dy)**2))
        # koeff2 = (K/(kon*N0)) * ((1/Lx*dx)**2)
        # koeff3 = (K/(kon*N0)) * ((1/Ly*dy)**2)
        
        diag = np.copy(bands[0]) #* koeff1
        supDiag = np.copy(bands[1]) #* koeff2
        subDiag = np.copy(bands[1]) #* koeff2
        superDiag = np.copy(bands[2]) #* koeff3
        suberDiag = np.copy(bands[2]) #* koeff3

        lap = sp.diags([diag,supDiag,subDiag,superDiag,suberDiag],[0,1,-1,M,-M], format  = 'csr') 
        vec = np.zeros(N*M,dtype = float)
    
        for i in totBound:
            vec *= 0
            vec[i] = 1
            lap[i] = sp.lil_array(vec)
            
        lap = alpha* lap/dx**2

        Fired = False
        def f(V,n_1,r_1):
            #V[0] = n[i+1]
            #V[1] = r[i+1]
            return r_1 - V[1] - dt*N0*kon*Tscale*V[0]*V[1] + dt*Tscale*koff*(1-V[1])

        def g(V,n_1,r_1):
            return n_1 - V[0] + dt*lap.dot(V[0]) - dt*(R0*kon*Tscale)*V[0]*V[1] + dt*(R0*koff*Tscale/N0)*(1-V[1])

        for i in range(steps-1):
            #implicit approach

            f1 = lambda P: [f(P,n[i],r[i]), g(P,n[i],r[i])] 
            vec = root(f1,[n[i],r[i]], method = "krylov", tol = 1e-6)
            vec = vec.x
        
            n[i+1] = vec[0]
            r[i+1] = vec[1]

            #Semi implicitt approach
            # vn = n[i] + dt*(R0*koff*Tscale/N0)*(1-r[i]) - dt*(R0*kon*Tscale)*r[i]*n[i]
            # n[i+1] = lin.spsolve(Nevro_block,vn)
            
            # r[i+1] = r[i] - dt*N0*kon*Tscale*n[i+1]*r[i] + dt*Tscale*koff*(1-r[i])
            #q[i+1] = q[i] + dt*N0*kon*Tscale*n[i+1]*r[i] - dt*Tscale*koff*q[i]
    
        return r,n

def initial(P, x = 1, y = 0, r = 0.1):
    arr = np.zeros(len(P))
    for i,points in enumerate(P):
        if (points[0] - 0.5)**2 + (points[1] - 0.5)**2 < r**2:
            arr[i] = 1
    
    return arr


if __name__ == "__main__":
    N = 50

    r0 = fatCircle(N)
    r0 = initial(r0)
    r0 = r0/np.sum(r0)

    n0 = fatCircle(N)
    n0 = initial(n0,r = 0.05)
    n0 = n0/np.sum(n0)


    kon = 4e3
    koff = 5
    K = 8*10**(-7)
    Times = 100

    DX = 1/N
    DY = 1/N
    DT = 0.0001

    rec, nevro = solver(N,r0, n0, Times, (DX,DY), DT, fat = True)

    bound = np.where(rec != 0, 1 - rec, 0)
    points = fatCircle(N)
    print(sum(nevro[-1]))
    print(sum(rec[-1]))

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
        ax.set_title(f"Time elapsed {i * DT:.8f}s")
        ax.plot_trisurf(points[:,0], points[:,1], nevro[i], cmap='inferno') 

    # Call animate method
    ani = animation.FuncAnimation(fig, animate, Times, interval = 10, blit=False)

    # Display the plot
    plt.show()