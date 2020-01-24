import numpy as np
import random 
from sympy import solve, symbols, pprint, diff
import sympy 
import helper 
import matplotlib.pyplot as plt
from findM2 import findM2_function
import visualize
 

"""
Homework4.
Replace 'pass' by your implementation.
"""

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
  
    x1 = pts1[:,0]/M
    y1 = pts1[:,1]/M


    x1_ = pts2[:,0]/M
    y1_ = pts2[:,1]/M
    
    U =np.vstack( ( x1*x1_, y1*x1_, x1_, x1*y1_, y1*y1_, y1_, x1, y1, np.ones((x1.shape[0])) ) )

    V = np.matmul(U,np.transpose(U))

    w, v = np.linalg.eig(V)

    e_vals, e_vecs = np.linalg.eig(V)  

    F = e_vecs[:, np.argmin(e_vals)] 
    
    F = np.reshape(F, (3,3))

    F = helper.refineF(F, pts1/M, pts2/M)

    F = np.transpose(T)@F@T
    
    return F

                 
'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
  
    x1 = pts1[:,0]/M
    y1 = pts1[:,1]/M


    x1_ = pts2[:,0]/M
    y1_ = pts2[:,1]/M
    
    U =np.vstack( ( x1*x1_, y1*x1_, x1_, x1*y1_, y1*y1_, y1_, x1, y1, np.ones((x1.shape[0])) ) )

    u,s,v = np.linalg.svd(U.T)
    f1 = v[8,:]
    f2 = v[7,:]

    F1 = np.reshape(f1, (3,3))
    F2 = np.reshape(f2, (3,3))

    l = symbols('l')
    
    F = (l)*F1 +(1-l)*F2
    
    F_ = sympy.Matrix(F)
    
    eq = F_.det()
    
    roots = solve(eq)

    roots = np.fromiter(roots, dtype=complex)
    
    Farray = []
    for i in range(0, len(roots)):
            x = roots[i]
            if x.imag ==0:
                x = x.real
                F = x*F1 + (1-x)*F2
                F = helper.refineF(F, pts1/M, pts2/M)
                F = np.transpose(T)@F@T
                Farray.append(F)
                
    Farray = np.array(Farray)
    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K1.T@F@K2
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    Ws = []
    for i in range(len(pts1)):
        u1 , v1 = pts1[i]
        u2 , v2 = pts2[i]

        d1 = u1*C1[2]-C1[0]
        d2 = v1*C1[2]-C1[1]
        d3 = u2*C2[2]-C2[0]
        d4 = v2*C2[2]-C2[1]
        
        D = np.vstack((d1,d2,d3,d4))

        V = np.matmul(np.transpose(D),D)

        w, v = np.linalg.eig(V)

        e_vals, e_vecs = np.linalg.eig(V)  

        W = e_vecs[:, np.argmin(e_vals)] 

        W = W/W[3]
        
        Ws.append(W)
        
    Ws = np.array(Ws)

    
    pts1_ = C1@Ws.T
    pts1_ = pts1_/pts1_[2]
    pts1_ = pts1_[0:2,:]
    pts1_ = pts1_.T
    
    pts2_ = C2@Ws.T
    pts2_ = pts2_/pts2_[2]
    pts2_ = pts2_[0:2,:]
    pts2_ = pts2_.T

    er2 = np.sum((pts2_-pts2)**2)
    er1 = np.sum((pts1_-pts1)**2)
    err = er1 + er2
    
    w = Ws[:,0:3]
    
    print("error",err)

    return [w, err]


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x2 , y2 , mini = 0,0,100
    Pl = np.array([x1,y1,1])
    Pl = np.expand_dims(Pl,axis=1)
    Pl = Pl/np.linalg.norm(Pl)
    temp = F@Pl

    x1 = int(round(x1))
    y1 = int(round(y1))
    size = 10

    window1 = im1[y1-size:y1+size+1, x1-size:x1+size+1]

    i=0
    for y in range(y1-size, y1+size):
        
        x = (y*temp[1]+temp[2])/(-temp[0])
        x = int(np.around(x))


        xx1 = y-size
        xx2 = y+size+1
        yy1 = x-size
        yy2 = x+size+1
        if yy1 > 0 and yy2 < im2.shape[1] and xx1 > 0 and xx2 < im2.shape[0]:
            window2 = im2[xx1:xx2, yy1:yy2]
            err = np.sum(np.linalg.norm(window2-window1)) 
                
            if i==0:
                mini = err
                x2   = x
                y2   = y

            if err<mini:
                mini = err
                x2   = x
                y2   = y
        i +=1

    return x2, y2


def BestFSevenPoints(pts1, pts2, M):
    tol = 0.001
    max_num = 0
    num_iter= 50
    bestF = []
    
    for itr in range(100):
        ind = np.random.randint(0,len(pts1),size=7)
        pts1_ = pts1[ind]
        pts2_ = pts2[ind]
        Farray = sevenpoint(pts1_, pts2_, M)
        for farray in range(Farray.shape[0]):
            p1 = np.array([pts1[:,0], pts1[:,1], np.ones(pts1.shape[0])])
            temp = Farray[farray,:,:]@p1
            temp = temp/np.linalg.norm(temp)
            p2 = np.array([pts2[:,0], pts2[:,1], np.ones(pts2.shape[0])])
            p2 = p2.T
            zeros = p2@temp
            zeros = zeros*np.identity(len(pts1))
            zeros = zeros[zeros!=0]
            num = np.sum(abs(zeros)<tol)
            if num>max_num:
                max_num = num 
                points1 = pts1_
                points2 = pts2_
                bestF = Farray[farray,:,:]
                Farray_best = Farray
                
    return bestF,Farray_best,points1,points2



if __name__ == "__main__":
    
    I1=plt.imread('../data/im1.png')
    I2=plt.imread('../data/im2.png')

    Ks= np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    
    M = np.max(I1.shape)
    
    data = np.load("../data/some_corresp.npz")
    pts1 = data['pts1']
    pts2 = data['pts2']
    
    
    
    data = np.load("../data/templeCoords.npz")
    x1 = data['x1']
    y1 = data['y1']
    
    
    F=eightpoint(pts1, pts2, M)
    
    np.savez_compressed('./q2_1.npz',F=F,M=M)
    
    helper.displayEpipolarF(I1, I2, F)
    
    bestF,Farray_best,points1,points2 = BestFSevenPoints(pts1, pts2, M)
    
    np.savez_compressed('./q2_2.npz',F=bestF, M=M, pts1=points1 , pts2=points2)
    
    helper.displayEpipolarF(I1, I2, bestF)

    E = essentialMatrix(F, K1, K2)
    
    Ms = helper.camera2(E)
    
    M2,M1,C2,C1,Points = findM2_function(K1,K2,Ms,pts1,pts2)

    np.savez_compressed('./q3_3.npz',M2=M2, C2=C2, P=Points)
    
    helper.epipolarMatchGUI(I1, I2, F)
    
    np.savez_compressed('./q4_1.npz',F=F, pts1=pts1 , pts2=pts2)
    
    visualize.Visualize(I1, I2, x1, y1,C1,C2,F)
    
    np.savez_compressed('./q4_2.npz',F=F, C1=C1,C2=C2,M2=M2,M1=M1)
    

    
    

    
    
    
