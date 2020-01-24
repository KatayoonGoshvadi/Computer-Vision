import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import *
import skimage.io

# write your script here, we recommend the above libraries for making your animation
data = np.load("../data/carseq.npy")
data = np.array(data)

rect = [59.0, 116.0, 145.0, 151.0]
rect = np.array(rect)
rect0 = np.copy(rect)
minx, miny, maxx, maxy = rect0[0],rect0[1],rect0[2],rect0[3]


p_sum = np.zeros((2,1))
T1 = data[:,:,0]

num_frames = data.shape[2]
next_Temp = True
thr = 0.1
rects = []
for i in range(0,num_frames-1):
    if(next_Temp):
        It = data[:,:,i]

    It1 = data[:,:,i+1]

    rect = np.array([minx, miny, maxx, maxy])
    rect = rect.reshape(4,1)
    rects.append(rect[:,0])
    
    pn = LucasKanade(It,It1,rect)
    
    p_sum += pn.reshape(2,1)

    pstar = LucasKanade(T1,It1,rect0,p_sum)

    if(np.linalg.norm(p_sum-pstar)<thr):
        p_sum = pstar
        minx = rect0[0] + p_sum[0,0] 
        maxx = rect0[2] + p_sum[0,0] 
        miny = rect0[1] + p_sum[1,0] 
        maxy = rect0[3] + p_sum[1,0] 
        change_Temp = 1
    else:
        p_sum = p_sum - pn
        next_Temp = False
        


rect = np.array([minx, miny, maxx, maxy])
rect = rect.reshape(4,1)
rects.append(rect[:,0])
np.save("carseqrects-wcrt",rects)

rectangles_LK = np.load('./carseqrects.npy')

for i in range(0,num_frames-1):
    It = data[:,:,i]
    rect = rects[i]
    if((i+1)==1 or (i+1)%100==0):
        fig,ax = plt.subplots(1)
        ax.imshow(It,cmap='gray')
        rect1 = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], fill=False, edgecolor='r', linewidth=2)
        ax.add_patch(rect1)
        rect2 = rectangles_LK[i]
        rect2 = patches.Rectangle((rect2[0], rect2[1]),rect2[2]-rect2[0],rect2[3]-rect2[1],fill=False, edgecolor='b', linewidth=2)
        ax.add_patch(rect2)
        plt.savefig("Template_correction"+str(i+1)+".png")
        plt.show()
