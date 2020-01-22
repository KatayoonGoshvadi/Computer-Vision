import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt 

def LucasKanadeBasis(It, It1, rect, bases):
    # Input: 
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	bases: [n, m, k] where nxm is the size of the template.
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    
    p = np.zeros(2)

    Iy, Ix = np.gradient(It1)

    w_rect = -rect[0]+rect[2]+1
    h_rect = -rect[1]+rect[3]+1
    
    h , w = It.shape
    X = range(h)
    Y = range(w)

    interSpline_Ix  = RectBivariateSpline(X,Y,Ix)
    interSpline_Iy  = RectBivariateSpline(X,Y,Iy)
    interSpline_It1 = RectBivariateSpline(X,Y,It1)
    interSpline = RectBivariateSpline(X,Y,It)
    
    x_inter = np.linspace(rect[0],rect[2],w_rect)
    y_inter = np.linspace(rect[1],rect[3],h_rect)


    orthobases = bases.reshape(-1,bases.shape[2])
    
    bases_sum = np.sum(orthobases.T@orthobases)
        
    
    thr = 0.1
    dp  = 1

    while np.linalg.norm(dp)> thr:
        
        x_inter_It1 = np.linspace(rect[0]+p[0],rect[2]+p[0],w_rect)
        y_inter_It1 = np.linspace(rect[1]+p[1],rect[3]+p[1],h_rect)
        
        Template = interSpline(y_inter,x_inter)
        
        It1_warped = interSpline_It1(y_inter_It1,x_inter_It1)
        
        error = Template- It1_warped
        error = error.reshape(-1,1)
        error = (1 - bases_sum) * error
        
        Ixx = interSpline_Ix(y_inter_It1,x_inter_It1)
        Iyy = interSpline_Iy(y_inter_It1,x_inter_It1)

        dI = np.hstack((Ixx.reshape(-1,1),Iyy.reshape(-1,1)))
    
        dw_p = np.array([[1,0],[0,1]])
        

        
        temp = np.matmul(dI,dw_p)
        temp = (1 - bases_sum) * temp
        temp_t = np.transpose(temp)
        
        H = np.matmul(temp_t,temp)
        dp = np.linalg.inv(H)@temp_t@error

        p[0] = p[0] + dp[0,0]
        p[1] = p[1] + dp[1,0]
        
    return p