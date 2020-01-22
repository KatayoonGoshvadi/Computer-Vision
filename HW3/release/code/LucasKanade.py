import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input: 
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    
    p = p0

    Iy, Ix = np.gradient(It1)

    w_rect = -rect[0]+rect[2]+1
    h_rect = -rect[1]+rect[3]+1
    
    h , w = It.shape
    X = range(h)
    Y = range(w)

    interSpline_Ix  = RectBivariateSpline(X,Y,Ix)
    interSpline_Iy  = RectBivariateSpline(X,Y,Iy)
    interSpline_It1 = RectBivariateSpline(X,Y,It1)


    x_inter = np.linspace(rect[0],rect[2],w_rect)
    y_inter = np.linspace(rect[1],rect[3],h_rect)

    interSpline = RectBivariateSpline(X,Y,It)
    Template = interSpline(y_inter,x_inter)


    thr = 0.1
    dp  = 1

    while np.linalg.norm(dp)> thr:
        x_inter_It1 = np.linspace(rect[0]+p[0],rect[2]+p[0],w_rect)
        y_inter_It1 = np.linspace(rect[1]+p[1],rect[3]+p[1],h_rect)
        
        It1_warped = interSpline_It1(y_inter_It1,x_inter_It1)
        
        error = Template- It1_warped
        
        error = error.reshape(-1,1)
    
        Ixx = interSpline_Ix(y_inter_It1,x_inter_It1)
        Iyy = interSpline_Iy(y_inter_It1,x_inter_It1)
 
        dI = np.hstack((Ixx.reshape(-1,1),Iyy.reshape(-1,1)))
        
        dw_p = np.array([[1,0],[0,1]])
        
        temp = np.matmul(dI,dw_p)
        temp_t = np.transpose(temp)
    
        H = np.matmul(temp_t,temp)
        
        dp = np.linalg.inv(H)@temp_t@error
        

        p[0] = p[0] + dp[0,0]
        p[1] = p[1] + dp[1,0]
        
    return p
