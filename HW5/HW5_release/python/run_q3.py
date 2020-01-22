import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 10
learning_rate = 0.008
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batches_valid = get_random_batches(valid_x,valid_y,batch_size)
batch_num = len(batches)

# print("batchesssssssssssssssssss=",np.shape(batches))
params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')

# with default settings, you should get loss < 150 and accuracy > 80%


train_acc_ls = []
valid_acc_ls = []

train_loss_ls = []
valid_loss_ls = []

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss = total_loss + loss
        total_acc = total_acc + acc


        delta1 = probs
        delta1[np.arange(probs.shape[0]),yb.argmax(axis=1)] -= 1

        delta2 = backwards(delta1,params,'output',linear_deriv)
        grad_X = backwards(delta2,params,'layer1',sigmoid_deriv)

        grad_W_l1 = params['grad_W' + 'layer1']
        grad_b_l1 = params['grad_b' + 'layer1']

        params['Wlayer1'] = params['Wlayer1'] - learning_rate*grad_W_l1
        params['blayer1'] = params['blayer1'] - learning_rate*grad_b_l1

        grad_W_o = params['grad_W' + 'output']
        grad_b_o = params['grad_b' + 'output']

        params['Woutput'] = params['Woutput'] - learning_rate*grad_W_o
        params['boutput'] = params['boutput'] - learning_rate*grad_b_o

    total_acc = total_acc/len(batches)
    train_acc_ls.append(total_acc)
    
    total_loss= total_loss/len(train_x)
    train_loss_ls.append(total_loss)
  
        
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(valid_y,probs)
    loss = loss/len(valid_x)
    
    valid_acc_ls.append(acc)
    valid_loss_ls.append(loss)
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc: {:.2f}".format(itr,total_loss,total_acc))
        print("itr: {:02d} \t loss_v: {:.2f} \t acc_v: {:.2f}".format(itr,loss,acc))




import matplotlib.pyplot as plt

plt.plot(range(len(valid_acc_ls)),valid_acc_ls)

plt.plot(range(len(train_acc_ls)),train_acc_ls)

plt.gca().legend(('Valid','Train'))
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()



plt.plot(range(len(valid_loss_ls)),valid_loss_ls)

plt.plot(range(len(train_loss_ls)),train_loss_ls)

plt.gca().legend(('Valid','Train'))
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
        
        
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# weights for trained
params = pickle.load(open('q3_weights.pickle', 'rb'))

fig = plt.figure(1)
weights = params['Wlayer1']
grid = ImageGrid(fig,111,nrows_ncols=(8, 8),axes_pad=0)


for i in range(8):
    for j in range(8):
        grid[8*i+j].imshow(weights[:,8*i+j].reshape(32,32))
plt.show()
    
    
# weights for initial params 
fig = plt.figure(1)
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
W_int= params['Wlayer1']
grid = ImageGrid(fig,111,nrows_ncols=(8, 8),axes_pad=0)


for i in range(8):
    for j in range(8):
        grid[8*i+j].imshow(W_int[:,8*i+j].reshape(32,32))
plt.show()
 
# # Q3.1.3
params = pickle.load(open('q3_weights.pickle', 'rb'))

# confusion matrix on test data

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))


h1 = forward(test_x,params,'layer1')
probs = forward(h1,params,'output',softmax)

for i in range(len(probs)):
    class_predict = np.argmax(probs[i])
    class_correct = np.argmax(test_y[i])
    if class_predict == class_correct:
        confusion_matrix[class_correct,class_correct] += 1
    else:
        confusion_matrix[class_correct,class_predict] += 1
        confusion_matrix[class_predict,class_correct] += 1
# print(confusion_matrix)

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()



# confusion matrix on train data

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))


h1 = forward(train_x,params,'layer1')
probs = forward(h1,params,'output',softmax)

for i in range(len(probs)):
    class_predict = np.argmax(probs[i])
    class_correct = np.argmax(train_y[i])
    if class_predict == class_correct:
        confusion_matrix[class_correct,class_correct] += 1
    else:
        confusion_matrix[class_correct,class_predict] += 1
        confusion_matrix[class_predict,class_correct] += 1
# print(confusion_matrix)

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
