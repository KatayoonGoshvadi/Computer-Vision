'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import submission as sub
from findM2 import findM2_function
from mpl_toolkits.mplot3d import Axes3D



def Visualize(I1, I2, x1, y1,C1,C2,F):
    x2s=[]
    y2s=[]
    for i in range(len(x1)):
        [x2,y2] =sub.epipolarCorrespondence(I1, I2, F, x1[i,0], y1[i,0])
        x2s.append(x2)
        y2s.append(y2)
    

    pts1 = np.stack((x1[:,0],y1[:,0]),axis=1)
    pts2 = np.stack((np.array(x2s),np.array(y2s)),axis=1)

    [w, err] = sub.triangulate(C1, pts1, C2, pts2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(w[:,0],w[:,1],w[:,2])
    plt.show()
