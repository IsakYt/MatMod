import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

K = 8e-7
dx = 0.1
dy = 0.1
dt = 0.1
kon = 4e6
koff = 5

r0 = 191
n0 = 5000
q0 = 0

def build(M,N):
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
    diag = -4*np.ones(N*M, dtype = float)
    offDiag = np.ones(N*M - 1, dtype = float)
    superDiag = np.ones(N*M - M, dtype = float)
    system = [diag, offDiag, superDiag]

    return system, boundary


def fatCircle(N):
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    h = x[1] - x[0]
    c = h*2**0.5
    xv,yv = np.meshgrid(x,y)
    index = []
    nodes = []
    ys = yv.flatten()

    for i, point in enumerate(xv.flatten()):
        index.append(i)
        nodes.append((point,ys[i]))

    #Circle interior
    circNode = []
    for j,p in enumerate(nodes):
        if (p[0]-0.5)**2  + (p[1]-0.5)**2 < (0.5 + c)**2:
            circNode.append(p)
    circNode = np.array(circNode)
   
    return circNode

def fatLap(N):
    grid = fatCircle(N)
    A = np.zeros((len(grid[:,0]),len(grid[:,0])))

    data = []
    row = []
    col = []

    h = 1/N
    h2 = h*h
    for i,P in enumerate(grid):
        if (P[0]-0.5)**2 + (P[1]-0.5)**2 >= 0.25:
            A[i,i] = 1

        else:
            Px = P[0]
            Py = P[1]
            
            So = np.where(np.all(np.isclose([Px,Py-h],grid),axis=1))[0][0]
            Ea = np.where(np.all(np.isclose([Px+h,Py],grid),axis=1))[0][0]
            No = np.where(np.all(np.isclose([Px,Py+h],grid),axis=1))[0][0]
            We = np.where(np.all(np.isclose([Px-h,Py],grid),axis=1))[0][0]
            
            A[i][i] = -4
            A[i][So] = 1
            A[i][Ea] = 1
            A[i][No] = 1
            A[i][We] = 1
         
    A = csr_matrix(A/h2)
    return A

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