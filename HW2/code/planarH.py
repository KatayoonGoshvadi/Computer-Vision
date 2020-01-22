import numpy as np
import cv2
from BRIEF import briefLite, briefMatch



def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    
    index = np.random.permutation(range(p1.shape[1]))
    indexes = index[0:4]
    p1 = p1[:,indexes]
    p2 = p2[:,indexes]
    H2to1 = None
    
    A = np.zeros((8,9))
    
    temp  = np.append(p2,np.ones((1,4)),axis=0)
    temp2 = np.append(-temp,np.zeros((3,4)),axis=0)
    temp3 = np.append(temp2,temp*p1[0],axis=0)
    
    temp2_ = np.append(np.zeros((3,4)),-temp,axis=0)
    temp3_= np.append(temp2_,temp*p1[1],axis=0)
    
    temp3 = np.transpose(temp3)
    temp3_= np.transpose(temp3_)
    
    A[range(0,8,2),:]=temp3
    A[range(1,8,2),:]=temp3_


    V = np.matmul(np.transpose(A),A)

    w, v = np.linalg.eig(V)
    
    e_vals, e_vecs = np.linalg.eig(V)  
    
    H = e_vecs[:, np.argmin(e_vals)] 
    
    H2to1 = np.reshape(H,(3,3))

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    bestH=None
    p1 = locs1[matches[:,0]][:,0:2]
    p1 = np.transpose(p1)
    p2 = locs2[matches[:,1]][:,0:2]
    p2 = np.transpose(p2)
    
    inliners_max = 0

    for iter in range(num_iter):#num_iter
        H = computeH(p1,p2)
        
        points1 = np.vstack((p1,np.ones((1,p1.shape[1]))))

        points2 = np.vstack((p2,np.ones((1,p2.shape[1]))))

        points1_= H@points2

        points1_ = points1_/points1_[2];

        dists = np.sqrt (np.sum( (np.transpose(points1)-np.transpose(points1_))**2,axis=1 ) )
        

        inliners = np.sum( dists < tol) 
        

        
        if iter==0:
            bestH = H
            inliners_max = inliners

        if inliners_max < inliners:
            inliners_max = inliners
            bestH = H

    
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH=ransacH(matches, locs1, locs2, num_iter=5000, tol=2)


