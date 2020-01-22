import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import *
from LucasKanade import *
import copy


# write your script here, we recommend the above libraries for making your animation

bases = np.load('../data/sylvbases.npy')
data = np.load('../data/sylvseq.npy')
num = data.shape[2]
rect = [101.0, 61.0, 155.0, 107.0]
rect = np.array(rect)

rect_LK = rect.copy()
w = rect[2]-rect[0]
h = rect[3]-rect[1]
rects = []

indexs = [1,200,300,350,400]

for i in range(0,num-1):
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    
    rect_ = np.copy(rect)
    rects.append(rect_)
    
    if (i + 1) in indexs:
        fig,ax = plt.subplots(1)
        ax.imshow(It1,cmap='gray')

        rects2 = patches.Rectangle((rect_LK[0],rect_LK[1]),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rects2)

        rects1 = patches.Rectangle((rect[0],rect[1]),w,h,linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rects1)
        plt.savefig("Sylv"+str(i+1)+".png")
        plt.show()
        
    p = LucasKanadeBasis(It, It1, rect, bases)
    rect[0] = rect[0]+p[0]
    rect[1] = rect[1]+p[1]
    rect[2] = rect[2]+p[0]
    rect[3] = rect[3]+p[1]
    
    p_LK = LucasKanade(It, It1, rect_LK)
    rect_LK[0] = rect_LK[0]+p_LK[0]
    rect_LK[1] = rect_LK[1]+p_LK[1]
    rect_LK[2] = rect_LK[2]+p_LK[0]
    rect_LK[3] = rect_LK[3]+p_LK[1]
    
rect_ = np.copy(rect)
rects.append(rect_)
rects = np.array(rects)
np.save('sylvseqrects.npy',rects)



