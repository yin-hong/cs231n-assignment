from data_utils import load_CIFAR10
import numpy as np
from linear_svm import svm_loss_naive
from linear_svm import svm_loss_vectorized
import time
from gradient_check import grad_check_sparse
from linear_classifier import  LinearSVM
import matplotlib.pyplot as plt
# Load the raw CIFAR-10 data.
cifar10_dir='/home/hongyin/file/cifar-10-batches-py'
X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)
print('Traing data shape: ',X_train.shape)
print('Traing labels shape: ',y_train.shape)
print('Test data shape: ',X_test.shape)
print('Test label shape: ',y_test.shape)


# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training=49000
num_validation=1000
num_test=1000
num_dev=500

# Our validation set will be num_validation points from the original
# training set.
mask=range(num_training,num_training+num_validation)
X_val=X_train[mask]
y_val=y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask=range(num_training)
X_train=X_train[mask]
y_train=y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask=np.random.choice(num_training,num_dev,replace=False)
X_dev=X_train[mask]
y_dev=y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask=range(num_test)
X_test=X_test[mask]
y_test=y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train=np.reshape(X_train,(X_train.shape[0],-1))
X_val=np.reshape(X_val,(X_val.shape[0],-1))
X_test=np.reshape(X_test,(X_test.shape[0],-1))
X_dev=np.reshape(X_dev,(X_dev.shape[0],-1))
# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image=np.mean(X_train,axis=0)
print(mean_image[:10])
# second: subtract the mean image from train and test data
X_train=X_train-mean_image
X_val=X_val-mean_image
X_test=X_test-mean_image
X_dev=X_dev-mean_image
# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train=np.hstack((X_train,np.ones((X_train.shape[0],1))))
X_val=np.hstack((X_val,np.ones((X_val.shape[0],1))))
X_test=np.hstack((X_test,np.ones((X_test.shape[0],1))))
X_dev=np.hstack((X_dev,np.ones((X_dev.shape[0],1))))
print(X_train.shape,X_val.shape,X_test.shape,X_dev.shape)

# generate a random SVM weight matrix of small numbers
W=np.random.randn(3073,10)*0.0001
loss,grad=svm_loss_naive(W,X_dev,y_dev,0.000005)
print('loss:%f'%(loss,))


# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

# Compute the loss and its gradient at W.
loss,grad=svm_loss_naive(W,X_dev,y_dev,0.0)
# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)


# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic=time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc=time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be much faster.
print('difference: %f' % (loss_naive - loss_vectorized))


print('make sure ve is right')

# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

# Compute the loss and its gradient at W.
loss,grad=svm_loss_vectorized(W,X_dev,y_dev,0.0)
# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
f = lambda w: svm_loss_vectorized(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_vectorized(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_vectorized(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)


# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
svm=LinearSVM()
tic=time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True)
toc=time.time()
print('That tooks %fs'%(toc-tic))

"""
# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.savefig('/home/hongyin/file/cs231n-assignment1/picFaster.jpg')
"""

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print('y_train shape',y_train.shape)
print('y_train_pred shape',y_train_pred.shape)
print('same',sum(y_train==y_train_pred))
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))



# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
print('tune hyperparameters')
print('')
print('')
learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
regularization_strengths = [(1+i*0.1)*1e4 for i in range(-3,3)] + [(2+0.1*i)*1e4 for i in range(-3,3)]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1  # The highest validation accuracy that we have seen so far.
best_svm = None  # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
for learning_rate in learning_rates:
    for regularization_strength in regularization_strengths:
        lb_svm = LinearSVM()
        lb_svm_loss=lb_svm.train(X_train,y_train,learning_rate,regularization_strength,num_iters=5000,batch_size=200,verbose=True)
        y_train_pred = lb_svm.predict(X_train)
        print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
        y_val_pred = lb_svm.predict(X_val)
        print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))
        tmp_val=np.mean(y_val == y_val_pred)
        if tmp_val>best_val:
            best_val=tmp_val
            best_svm=lb_svm

################################################################################
#                              END OF YOUR CODE                                #
################################################################################


for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)