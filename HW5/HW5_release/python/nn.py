import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    temp = np.sqrt(6/(in_size+out_size)) 
    W = np.random.uniform(-temp,temp,(in_size, out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res


# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    
#     print("XXXXX=",np.shape(X))
#     print("WWWWW=",np.shape(W))
#     print("bbbbb=",np.shape(b))
   
    pre_act = X@W+b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    x   = x-np.expand_dims(np.amax(x,axis=1),axis=1)
    res = (np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=1),axis=1))
    return res


# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    probs_y_ind = np.argmax(probs,axis=1)
    y_ones_index = np.argmax(y,axis=1)
    acc = np.sum(y_ones_index==probs_y_ind)/y.shape[0]
    loss = - np.sum(y*np.log(probs))
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    act_d = activation_deriv(post_act)
    grad_X= delta*act_d@W.T
    grad_W= X.T@(delta*act_d)
    grad_b= np.sum(delta*act_d,axis = 0)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X


# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    num_batch = int(x.shape[0]/batch_size)
    indexes = np.split(index,num_batch)
    for i in range(num_batch):
        new_batch=[x[indexes[i]],y[indexes[i]]]
        batches.append(new_batch)
    return batches