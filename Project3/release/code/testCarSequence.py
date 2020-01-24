import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation
rects = []
data = np.load('../data/carseq.npy')
num_frames = data.shape[2]
rect = [59.0, 116.0, 145.0, 151.0]
rect = np.array(rect)
w = rect[2]-rect[0]
h = rect[3]-rect[1]

rect_=np.copy(rect)
rects.append(rect_)
for i in range(0,num_frames-1):
    It = data[:,:,i]
    It1=data[:,:,i+1]
    
    if (i+1)==1 or (i+1)%100==0:
        fig,ax = plt.subplots(1)
        ax.imshow(It,cmap='gray')
        rects_plot = patches.Rectangle((rect[0],rect[1]),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rects_plot)
        plt.savefig("CarSeq"+str(i+1)+".png")
        plt.show()

    p = LucasKanade(It,It1, rect)
    rect[0] = rect[0]+p[0]
    rect[1] = rect[1]+p[1]
    rect[2] = rect[2]+p[0]
    rect[3] = rect[3]+p[1]
    rect_=np.copy(rect)
    rects.append(rect_)
       
rects = np.array(rects)
np.save('./carseqrects.npy',rects)

