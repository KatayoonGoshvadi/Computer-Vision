import numpy as np
import cv2
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = gaussian_pyramid[:,:,1:]-gaussian_pyramid[:,:,:-1]
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    Dx  =cv2.Sobel(DoG_pyramid,cv2.CV_64F,1,0,ksize=3)
    Dxx =cv2.Sobel(Dx ,cv2.CV_64F,1,0,ksize=3) 
    Dy  =cv2.Sobel(DoG_pyramid,cv2.CV_64F,0,1,ksize=3)
    Dyy =cv2.Sobel(Dy ,cv2.CV_64F,0,1,ksize=3)
    Dxy =cv2.Sobel(Dx ,cv2.CV_64F,0,1,ksize=3) 
    Dyx =cv2.Sobel(Dy ,cv2.CV_64F,1,0,ksize=3)
    
    h1 = np.stack((Dxx,Dxy),axis=3)
    h2 = np.stack((Dyx,Dyy),axis=3)
    h3 = np.stack((h1,h2),axis=3)

    Tr = np.trace(h3,axis1=3,axis2=4)
    
    Det= Dxx*Dyy-Dxy*Dyx
    
    Det_zeros = np.where(Det==0)
    
    Det_zeros = np.array(Det_zeros)
    
    principal_curvature = np.divide(Tr**2,Det)
    
    principal_curvature[Det_zeros[0],Det_zeros[1],Det_zeros[2]]=1000

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    print(np.shape(principal_curvature))
    print(np.shape(DoG_pyramid))
    print(np.shape(DoG_levels))
    index_r = np.where(principal_curvature < th_r )
    index_r = np.array(index_r)
    print("###",np.shape(index_r))
    DoG_pyramid_new = DoG_pyramid[index_r[0,:],index_r[1,:],index_r[2,:]]
    print("***",np.shape(DoG_pyramid))
    index_contrast = np.where(DoG_pyramid_new > th_contrast)
    index_contrast = np.squeeze(index_contrast,axis=0)
    result = index_r[:,index_contrast]
    result = np.transpose(result)
    locsDoG = np.column_stack((result[:,1],result[:,0]))
    locsDoG = np.column_stack((locsDoG,result[:,2]))
    print(np.shape(locsDoG))
    print(locsDoG)
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.04, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid , levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
# #     displayPyramid(im_pyr)
    
# #     # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
# # #     displayPyramid(DoG_pyr)
# #     # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
# #     displayPyramid(pc_curvature)
# #     # test get local extrema
#     th_contrast = 0.03
#     th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature)
    
#     print(np.shape(locsDoG) )
    # test DoG detector
#     locsDoG, gaussian_pyramid = DoGdetector(im)
    
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
#     plt1 = plt.imshow(im,cmap = plt.get_cmap('gray'))
    
#     plt2 = plt.scatter(locsDoG[:,0],locsDoG[:,1],c='r')

#     plt.show()

    
    


