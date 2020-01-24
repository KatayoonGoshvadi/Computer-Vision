import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt
import math
import scipy

from BRIEF import briefLite , briefMatch , plotMatches



im = cv2.imread('../data/model_chickenbroth.jpg')
locs1, desc1 = briefLite(im)
rows,cols,z = im.shape

corrects=[]
for theta in range(0,370,10):
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    im2 = cv2.warpAffine(im,M,(cols,rows))
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    
    angle = math.radians(theta)
    M = np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
    p1 = locs1[matches[:,0]][:,0:2]
    p2_correct = M@(np.transpose(p1)- np.array([[cols/2],[rows/2]])*np.ones((2,len(p1))))+ np.array([[cols/2],[rows/2]])*np.ones((2,len(p1)))
    p2 = locs2[matches[:,1]][:,0:2]
    p2=np.transpose(p2)
    
    plt1 = plt.imshow(im2,cmap = plt.get_cmap('gray'))
    plt2 = plt.scatter( p2_correct[0,:],p2_correct[1,:],c='r')
    plt2 = plt.scatter( p2[0,:],p2[1,:],c='y')
    plt.gca().legend(("correct match point","computed match point"))
    plt.show()

    dist=np.sqrt(np.sum((p2_correct-p2)**2,axis=0))
    correct = np.sum(dist<2)
    corrects.append(correct)
#     plotMatches(im,im2,matches,locs1,locs2)
#     plt.show()


objects = range(0,370,10)
y_pos = np.arange(len(objects))
performance = corrects
 
    
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Correct Matches')
plt.xlabel('Rotation Deg')
plt.title('Correct Matches VS Degree of Rotation')
plt.show()
