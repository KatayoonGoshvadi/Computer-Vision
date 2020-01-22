import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
from collections import Counter
import skimage.io



def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    SPM_layer_num = 3
    features = []
    i = 0
    for path in train_data['files']:
        features.append(get_image_feature('../data/'+path,dictionary,SPM_layer_num,dictionary.shape[0]))
    features = np.asarray(features)
    np.savez_compressed('./trained_system',dictionary=dictionary,features=features,labels=train_data['labels'],SPM_layer_num=SPM_layer_num)


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")#havaaset bashe ino test koniiiii
    test_paths = test_data['files']
    test_labels = test_data['labels']

    trained_system = np.load("trained_system.npz")


    features = trained_system['features']
    train_labels   = trained_system['labels']
    SPM_layer_num = trained_system['SPM_layer_num']
    dictionary = trained_system['dictionary']

    confusion_matrix = np.zeros((8,8))
    correct_pred=0
    wrong_pred=0
    test_n=0
    for path in test_paths:
        test_feature = get_image_feature('../data/'+path,dictionary,SPM_layer_num,dictionary.shape[0])

        distances = distance_to_set(test_feature,features)

        closest_index = np.argmax(distances)

        if(train_labels[closest_index] == test_labels[test_n]):
            confusion_matrix[test_labels[test_n]][test_labels[test_n]] += 1
            correct_pred += 1
        else:
            confusion_matrix[test_labels[test_n]][train_labels[closest_index]] +=1
            wrong_pred += 1
            
#         print("pred=",train_labels[closest_index],"label",test_labels[test_n])
        test_n += 1
        
    accuracy= correct_pred/(correct_pred+wrong_pred)

    return confusion_matrix,accuracy

def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    path_img =file_path
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image,dictionary)
    hist_all= get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    return hist_all



def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    return np.sum((word_hist<=histograms)*word_hist + (word_hist > histograms)*histograms,axis=1)
    



def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    hist = np.histogram(wordmap,bins=range(dict_size+1),density=True)
    return hist[0]


def split_in_four( array2D):
    x , y = np.shape(array2D)
    return array2D[0:int(x/2),0:int(y/2)],array2D[0:int(x/2),int(y/2):],array2D[int(x/2):,0:int(y/2)],array2D[int(x/2):,int(y/2):]


def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    L = layer_num-1
    hist_all = [] 
    UL_l1 , UR_l1 , DL_l1 , DR_l1 = split_in_four(wordmap)
    UL_l2 = split_in_four(UL_l1)
    UR_l2 = split_in_four(UR_l1)
    DL_l2 = split_in_four(DL_l1)
    DR_l2 = split_in_four(DR_l1)
    
    
    for subimage in range(4):
        hist1 = get_feature_from_wordmap( UL_l2[subimage],dict_size)
        hist2 = get_feature_from_wordmap( UR_l2[subimage],dict_size)
        hist3 = get_feature_from_wordmap( DL_l2[subimage],dict_size)
        hist4 = get_feature_from_wordmap( DR_l2[subimage],dict_size)
        hist_all.append((0.5)*hist1)#(0.5)*
        hist_all.append((0.5)*hist2)
        hist_all.append((0.5)*hist3)
        hist_all.append((0.5)*hist4)
        
        
    hist_all.append((0.25)*get_feature_from_wordmap( UL_l1,dict_size))
    hist_all.append((0.25)*get_feature_from_wordmap( UR_l1,dict_size))
    hist_all.append((0.25)*get_feature_from_wordmap( DL_l1,dict_size))
    hist_all.append((0.25)*get_feature_from_wordmap( DR_l1,dict_size))

    hist_all.append((0.25)*get_feature_from_wordmap(wordmap,dict_size))
    
    hist_all = np.asarray(hist_all)
    
    hist_all = np.reshape(hist_all, hist_all.shape[0]*hist_all.shape[1])
    
    return hist_all
  








    

