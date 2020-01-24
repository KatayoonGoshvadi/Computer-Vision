import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import matplotlib.pyplot as plt
import cv2
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def SubtractDominantMotion(image1, image2):
    # Input:
    #	Images at time t and t+1 
    # Output:
    #	mask: [nxm]
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    h,w = image2.shape
    
    M =  LucasKanadeAffine(image1,image2)
    
#     M = InverseCompositionAffine(image1,image2)
    im2_w = cv2.warpAffine(image1,M,(w,h))
    im2_w = binary_erosion(im2_w)
    im2_w = binary_dilation(im2_w)
    mask = abs(im2_w -image1) 

    thr = 0.75

    mask = mask>thr

    return mask





