import numpy as np
import matplotlib.pyplot as plt


K = 8e-7
dx = 0.1
dy = 0.1
dt = 0.1
kon = 4e6
koff = 5

r0 = 191
n0 = 5000
q0 = 0

def build(M,N, banded = True):
    """
    Builds a system matrix based on 2D central differences for an MxN grid, assumes dx = dy.
    input:
        M: int, number of points along the x-axis
        N: int, number of points along the y-axis
    return:
        system: 2D array, the system matrix
        boundary: 1D array, lists the rows in 'system' which handle the boundary elements
    """
    #system = np.zeros((M*N, M*N),dtype = float)
    bottom = [i for i in range(M)] 
    west =  [k for k in range(M,   N*M - M,    M)] 
    east =  [m for m in range(2*M-1,   N*M - M+1,    M)]
    top =  [j for j in range(N*M - M, N*M)]
    boundary = [bottom, west, east, top]
    #print(boundary)

    #banded approach:
    if banded:
        diag = -4*np.ones(N*M, dtype = float)
        offDiag = np.ones(N*M - 1, dtype = float)
        superDiag = np.ones(N*M - M, dtype = float)

        system = [diag, offDiag, superDiag]

        return system, boundary

    else:
        for i in range(M,M*N-M):
            system[i,i] = -4
            system[i,i-1] = 1
            system[i,i+1] = 1
            system[i,i+M] = 1
            system[i,i-M] = 1    
    
        return system, boundary

#if __name__ == "__main__":
    # A = build(10,10)[0]
    # plt.imshow(A)
    # plt.show()

    # x = np.linspace(0,1,16)
    # y = np.copy(x)
    # xv,yv = np.meshgrid(x,y)

    # def test(x,y):
    #     return np.exp(-100*((x-0.5)**2 + (y-0.5)**2))

    # y = test(xv,yv)
    # Dy,bnd = laplace(y)

    # plt.contourf(xv,yv,y, cmap = "inferno")
    # plt.show()

    # plt.imshow(Dy,cmap = "inferno")
    # plt.show()