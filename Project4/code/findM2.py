# '''
# Q3.3:
#     1. Load point correspondences
#     2. Obtain the correct M2
#     3. Save the correct M2, C2, and P to q3_3.npz
# '''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def findM2_function(K1,K2,Ms,pts1,pts2):
    import submission as sub
    for i in range(4):
        M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        M2 = Ms[:,:,i]
        C1 = K1@M1
        C2 = K2@M2

        [w, err] = sub.triangulate(C1, pts1, C2, pts2)
        index = np.where(w[:,2]>0)
        index = np.array(index)
        if(index.shape[1] == w.shape[0] ):
            correct_M = Ms[:,:,2]
            correct_C2 = C2
            Points = w
        
    return correct_M,M1,correct_C2,C1,Points

