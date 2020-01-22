import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
    # Input: 
    #	It: template image
    #	It1: Current image

    # Output:
    #	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)

    h , w = It.shape

    X = range(h)
    Y = range(w)

    Iy, Ix = np.gradient(It)

    interSpline_Ix  = RectBivariateSpline(X,Y,Ix)
    interSpline_Iy  = RectBivariateSpline(X,Y,Iy)
    interSpline_It1 = RectBivariateSpline(X,Y,It1)
    interSpline = RectBivariateSpline(X,Y,It)

    X_mesh , Y_mesh = np.meshgrid(Y, X)
    x = np.reshape(X_mesh, (-1,1))
    y = np.reshape(Y_mesh, (-1,1))

    
    Ixx = interSpline_Ix(X,Y)
    Iyy = interSpline_Iy(X,Y)
    Template = interSpline(X,Y)
    
    Ixx = Ixx.reshape(-1,1)
    Iyy = Iyy.reshape(-1,1)
    Template = Template.reshape(-1)
    
    temp = np.stack((y*Iyy,x*Iyy,Iyy,y*Ixx,x*Ixx,Ixx),axis=1)
    temp = np.squeeze(temp,axis=2)
    O = np.ones((x.shape[0],1))
    It_coords = np.hstack((y,x,O))
    It_coords = np.transpose(It_coords)
    
    thr = 0.1
    dp  = 10
    while np.linalg.norm(dp)> thr:
        
        W = M + p.reshape(2,3)

        It1_coords = W@It_coords
        
        #removing coordinates out of common region
        width_check = np.logical_and(It1_coords[1]>=0, It1_coords[1]<w)
        height_check = np.logical_and(It1_coords[0]>=0, It1_coords[0]<h)
        correct_cords = np.logical_and(width_check,height_check)
        
        correct_cords = correct_cords.nonzero()

        correct_cords = correct_cords[0]

        
        It1_x = It1_coords[0,correct_cords]
        It1_y = It1_coords[1,correct_cords]
        It1_warped = interSpline_It1.ev(It1_x,It1_y)


        Template_correct = Template[correct_cords]
        error = Template_correct - It1_warped
        error = error.reshape(error.shape[0],1)

        temp2 = np.matmul(temp.T[:,correct_cords],error)


        H  = temp[correct_cords,:].T@temp[correct_cords,:]

        dp = np.linalg.pinv(H)@temp2

        p[0] += dp[0,0]
        p[1] += dp[1,0]
        p[2] += dp[2,0]
        p[3] += dp[3,0]
        p[4] += dp[4,0]
        p[5] += dp[5,0]

    M =  M + p.reshape(2,3) 
    return M






