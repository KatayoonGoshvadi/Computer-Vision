import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import scipy.misc


def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    filtered_image = extract_filter_responses(image)
    filtered_image_flatten = np.reshape(filtered_image,(filtered_image.shape[0]*filtered_image.shape[1],60))
    dist=scipy.spatial.distance.cdist(filtered_image_flatten,dictionary)
    wordmap = (np.reshape(np.argmin(dist,axis=1),(image.shape[0],image.shape[1])))#/dictionary.shape[0]
    
    
#     #just for three images in write up
#     np.save('./wordmap.npy',wordmap)
#     np.save('./wordmap{}.npy'.format(i),wordmap)
#     directory='./Wordmaps/'
#     if not os.path.exists(directory):
#         print("not exist")
#         os.makedirs(directory)
#     np.save('./Wordmaps/Image{}.npy'.format(i),filter_responses)


    
    return wordmap






def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    i,alpha,image_path = args
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255

    filtered_image = extract_filter_responses(image)

    filtered_image_flatten = np.reshape(filtered_image,(filtered_image.shape[0]*filtered_image.shape[1],60))

    index = np.random.permutation(range(len(filtered_image_flatten)))

    index_alpha=index[:alpha]

    filter_responses=filtered_image_flatten[index_alpha]

    directory='./Filter_responses/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('./Filter_responses/Image{}.npy'.format(i),filter_responses)


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    train_data = np.load("../data/train_data.npz")
    paths= train_data['files'] 
    i=0
    for path in paths:
        compute_dictionary_one_image([i,250,'../data/'+path])#alpha=50
        i += 1


    responses=[]       
    for j in range(i):
        filter_responses=np.load('./Filter_responses/Image{}.npy'.format(j))
        responses.append(filter_responses)
        

    x,y,z=np.shape(responses)
    kmean_in=np.reshape(responses,(x*y,z))
    kmeans = sklearn.cluster.KMeans(200,n_jobs=num_workers).fit(kmean_in)
    
    dictionary = kmeans.cluster_centers_
    np.save('./dictionary.npy',dictionary)