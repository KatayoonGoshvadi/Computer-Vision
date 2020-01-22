import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pickle
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw,cmap='gray_r')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
#     plt.show()
    
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    bboxes = np.array(bboxes)
    
    
    rows = []
    row = []
    batch = []
    
    for i in range(bboxes.shape[0] - 1):
        if abs(bboxes[i][0] - bboxes[i+1][0]) < 100:
            row.append(bboxes[i])
        else:
            row.append(bboxes[i])
            rows.append(row)
            row = []
    if i == bboxes.shape[0] - 2:
        row.append(bboxes[i+1])
        rows.append(row)

    rows = np.array(rows)

    for i in range(0,rows.shape[0]):
        row = np.array(rows[i])
        row = row[row[:, 1].argsort()]
        
        for j in range(row.shape[0]):
        
            up_y,up_x,down_y,down_x = row[j]
        
            center_x = int((down_x+up_x)/2)
            center_y = int((down_y+up_y)/2)

            len_box = max(down_y-up_y,down_x-up_x)

            up_x,down_x,up_y,down_y = center_x-int(len_box/2),center_x+int(len_box/2),center_y-int(len_box/2),center_y+int(len_box/2)

            cropped_img = bw[up_y:down_y,up_x:down_x]
            
            cropped_img = skimage.filters.gaussian(cropped_img)
            
            cropped_img = skimage.transform.resize(cropped_img, (28, 28))
            
            padded_img = 1-np.pad(cropped_img, 2, 'constant')
            
            img_f = padded_img.T
            
            batch.append(img_f.flatten())
            
    h1 = forward(batch,params,'layer1')
    probs = forward(h1,params,'output',softmax) 
    predicts = np.argmax(probs,axis=1)

    k = 0
    for row in range(rows.shape[0]):
        for col in range(len(rows[row])):
            print(letters[predicts[col+k]],end=" ")
        print()
        k += len(rows[row])
    print()