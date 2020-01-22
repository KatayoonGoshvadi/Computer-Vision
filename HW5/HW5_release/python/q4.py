import numpy as np

import skimage
import skimage.measure 
import skimage.color 
import skimage.restoration
import skimage.filters
import skimage.morphology 
import skimage.segmentation 

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # estimating noise
    noise_es = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    # Denoise 
    filtered_image = skimage.restoration.denoise_bilateral(image, sigma_color=noise_es)    
    # greyscale
    bw = skimage.color.rgb2grey(filtered_image)
    # threshold
    thre = skimage.filters.threshold_otsu(bw)
    bw = bw < thre
    # morphology
    bw = skimage.morphology.dilation(bw, skimage.morphology .square(7))
    bw = skimage.morphology.erosion(bw, skimage.morphology .square(3))
    #label
    label_image = skimage.measure.label(bw, neighbors=8, background=0.0)
    #Return an RGB image where color-coded labels are painted over the image.
    image_label_overlay = skimage.color.label2rgb(label_image, image = bw)

    bboxes = []
    for region in skimage.measure.regionprops(label_image):
        if region.area >= 800:
            y1, x1, y2, x2 = region.bbox
            bboxes.append(np.array([y1, x1, y2, x2]))
    
    return bboxes, bw