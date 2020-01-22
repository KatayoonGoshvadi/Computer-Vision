import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt
import os

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
        
    rows1,cols1,z = im1.shape   
    rows2,cols2,z = im2.shape
    
    corner_l_u = np.array([[0],[0],[1]])
    corner_r_u = np.array([[cols2],[0],[1]])
    corner_r_d = np.array([[cols2],[rows2],[1]])
    corner_l_d = np.array([[0],[rows2],[1]])

    corner_l_u = H2to1 @ corner_l_u
    corner_r_u = H2to1 @ corner_r_u
    corner_r_d = H2to1 @ corner_r_d
    corner_l_d = H2to1 @ corner_l_d
    

    corner_l_u = corner_l_u/corner_l_u[2]
    corner_r_u = corner_r_u/corner_r_u[2]
    corner_r_d = corner_r_d/corner_r_d[2]
    corner_l_d = corner_l_d/corner_l_d[2]
    
    
    left = int(min(corner_l_u[0], corner_l_d[0], 0))
    right = int(max(corner_r_u[0], corner_r_d[0], cols1))
    width = right - left
  
    up = int(min(corner_l_u[1],corner_r_u[1], 0))
    down = int(max(corner_l_d[1],corner_r_d[1], rows1))
    height = down - up

    
    pano_im = cv2.warpPerspective(im2, H2to1, (width,height))
    
#     cv2.imshow("Warped Image",pano_im)
#     cv2.waitKey(0)
    
    cv2.imwrite('../results/6_1.jpg',pano_im)

    np.save('../results/q6_1.npy',H2to1)
    
    pano_im[0:rows1,0:cols1] = im1
    
#     cv2.imwrite('../results/clipped.jpg',pano_im)

    
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    rows1,cols1,z = im1.shape   
    rows2,cols2,z = im2.shape
    
    corner_l_u = np.array([[0],[0],[1]])
    corner_r_u = np.array([[cols2],[0],[1]])
    corner_r_d = np.array([[cols2],[rows2],[1]])
    corner_l_d = np.array([[0],[rows2],[1]])

    corner_l_u = H2to1 @ corner_l_u
    corner_r_u = H2to1 @ corner_r_u
    corner_r_d = H2to1 @ corner_r_d
    corner_l_d = H2to1 @ corner_l_d
    

    corner_l_u = corner_l_u/corner_l_u[2]
    corner_r_u = corner_r_u/corner_r_u[2]
    corner_r_d = corner_r_d/corner_r_d[2]
    corner_l_d = corner_l_d/corner_l_d[2]
    
    
    left = int(min(corner_l_u[0], corner_l_d[0], 0))
    right = int(max(corner_r_u[0], corner_r_d[0], cols1))
    width = right - left
  
    up = int(min(corner_l_u[1],corner_r_u[1], 0))
    down = int(max(corner_l_d[1],corner_r_d[1], rows1))
    height = down - up

    tx = -int(min(corner_l_u[0], corner_l_d[0], 0))
    ty = -int(min(corner_l_u[1], corner_r_u[1], 0))
    
    M = np.float32([[1,0,tx],[0,1,ty],[0,0,1]])
    
    H2to1_M = M @ H2to1
    
    warp_im1= cv2.warpPerspective(im1, M, (width,height))
    
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), (width,height))
    

    pano_im = np.maximum(warp_im1,warp_im2)
    
#     cv2.imshow("Pano",pano_im)
#     cv2.waitKey(0)
    
    cv2.imwrite('../results/q6_1_pan.jpg',pano_im)
    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    cv2.imwrite('../results/q6_3.jpg', pano_im)



if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()