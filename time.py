from RHS import build
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solver(r0, n0, q0, steps, DX, dt):
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
    M, N = np.shape(r0)
    N0 = 5000
    R0 = 152
    r = np.zeros((steps,M*N),dtype = float)
    n = np.copy(r)
    q = np.copy(r)

    r[0] = r0.flatten()
    n[0] = n0.flatten()
    q[0] = q0.flatten()

    bands, boundary = build(M,N)

    bottom = boundary[0]
    west = boundary[1]
    east = boundary[2]
    top = boundary[3]

    #Use boundary vector to remove boundary indexes 
    #from n,r,q-vector?
    dx,dy = DX[0], DX[1]
    Lx = 22e-8
    Ly = 15e-9
    koeff1 = (K/(kon*N0)) * ((1/Lx*dx)**2 + (1/(Ly*dy)**2))
    koeff2 = (K/(kon*N0)) * ((1/Lx*dx)**2)
    koeff3 = (K/(kon*N0)) * ((1/Ly*dy)**2)
    
    diag = np.copy(bands[0]) #* koeff1
    supDiag = np.copy(bands[1]) #* koeff2
    subDiag = np.copy(bands[1]) #* koeff2
    superDiag = np.copy(bands[2]) #* koeff3
    suberDiag = np.copy(bands[2]) #* koeff3

    lap = sp.diags([diag,supDiag,subDiag,superDiag,suberDiag],[0,1,-1,M,-M], format  = 'csr')
    vec = np.zeros(N*M,dtype = float)
    
    for i in bottom:
        vec *= 0
        vec[i] = -1
        vec[i+M] = 1
        lap[i] = sp.lil_array(vec)

    for i in top:
        vec *= 0
        vec[i] = -1
        vec[i-M] = 1
        lap[i] = sp.lil_array(vec)

    for i in west:
        vec *= 0
        #vec[i] = -1   #Neumann
        #vec[i+1] = 1  #Neumann
        vec[i] = 1  #Dirichlet
        lap[i] = sp.lil_array(vec)

    for i in east:
        vec *= 0
        #vec[i] = -1  #Neumann
        #vec[i-1] = 1 #Neumann
        vec[i] = 1 #Dirichlet
        lap[i] = sp.lil_array(vec)

    lap = lap/dx**2
    

    #We need a block matrix which is (3*N*M)^2 big
    #The block order is r, n, q
    I = sp.diags([np.ones(N*M)],[0])
    Nevro_block = I - dt * lap
    for i in range(steps-1):
        
        vn = n[i] + dt*(R0*koff/(kon*N0**2))*q[i] - dt*(R0/N0)*r[i]*n[i]
        n[i+1] = lin.spsolve(Nevro_block,vn)
        
        #use the whole grid or just a subset?
        r[i+1] = r[i] - dt*n[i+1]*r[i] + dt*(koff/(kon*N0))*q[i]
        q[i+1] = q[i] + dt*r[i+1]*n[i+1] - dt*(koff/(kon*N0))*q[i] 

    return r,n,q

if __name__ == "__main__":
    M,N = 25,25
    #r and q should have initial values only in the last row
    #While n should have initial values only in the first row
    r0 = np.zeros((M,N))
    r0[-1][5:15] = 0.1
    n0 = np.zeros((M,N))
    n0[0][5:15] = 0.5

    q0 = np.zeros((M,N))
    kon = 4e3
    koff = 5
    K = 8e-6
    Times = 200
    Lx = 1
    Ly = 1
    DX = Lx/M
    DY = Ly/N
    DT = 0.0001

    rec, nevro, bound = solver(r0, n0, q0, Times, (DX,DY), DT)
    print(sum(nevro[-1]))
    print(sum(bound[-1]))
    print(bound[-1])
    x = np.linspace(0,1,M)
    y = np.linspace(0,1,N)
    xv,yv = np.meshgrid(x,y)
    #Here comes some "borrowed" code for animation:
    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Create a figure and a set of subplots
    fig = plt.figure()
    ax = plt.subplot(projection = '3d')

    # Method to change the contour data points
    def animate(i):
        ax.clear()
        #ax.set_zlim(np.min(nevro[i]),np.max(nevro[i]))
        ax.set_title(f"Time elapsed {i*DT:.4f}")
        ax.plot_surface(xv, yv, nevro[i].reshape(M,N), cmap='inferno') 

    # Call animate method
    ani = animation.FuncAnimation(fig, animate, Times, interval = 100, blit=False)

    # Display the plot
    plt.show()

    