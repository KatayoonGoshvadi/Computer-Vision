import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import *
from LucasKanadeAffine import *
from InverseCompositionAffine import InverseCompositionAffine

# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/aerialseq.npy')
num_ = frames.shape[2]
for i in range(num_-1):
    image1 = frames[:, :, i]
    image2 = frames[:, :, i+1]
    mask = SubtractDominantMotion(image1, image2)
    index = np.where(mask==1)
    index = np.array(index)
    j = i+1
    if j == 30 or j == 60 or j == 90 or j == 120:
        plt.imshow(image2,cmap='gray')
        plt.plot(index[1],index[0],'b.')
        plt.savefig("air"+str(i+1)+"png")
        plt.show()

