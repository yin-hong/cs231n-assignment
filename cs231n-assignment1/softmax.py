from data_utils import load_CIFAR10
import numpy as np
from softmax_classfier import softmax_loss_naive
import time
from gradient_check import grad_check_sparse
from softmax_classfier import softmax_loss_vectorized
from linear_classifier import Softmax
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

# First implement the naive softmax loss function with nested loops.
# Open the file cs231n/classifiers/softmax.py and implement the
# softmax_loss_naive function.


# Generate a random softmax weight matrix and use it to compute the loss.
W=np.random.randn(3073,10)*0.0001
loss,grad=softmax_loss_naive(W,X_dev,y_dev,0.0)
# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# similar to SVM case, do another gradient check with regularization
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)

"""
print('Make sure vectorized is right')
print('')
print('')
print('')
# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
loss, grad = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
f = lambda w: softmax_loss_vectorized(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# similar to SVM case, do another gradient check with regularization
loss, grad = softmax_loss_vectorized(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_vectorized(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)
"""
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
results={}
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths =[(1+0.1*i)*1e4 for i in range(-3,4)] + [(5+0.1*i)*1e4 for i in range(-3,4)]
################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
for learning_rate in learning_rates:
    for regularization_strength in regularization_strengths:
        softmax_tmp=Softmax()
        softmax_tmp.train(X_train,y_train,learning_rate,regularization_strength,num_iters=2000)
        y_train_pred=softmax_tmp.predict(X_train)
        print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
        tmp_train=np.mean(y_train == y_train_pred)
        y_val_pred=softmax_tmp.predict(X_val)
        print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))
        tmp_val=np.mean(y_val == y_val_pred)
        results[(learning_rate,regularization_strength)]=(tmp_train,tmp_val)
        if tmp_val>best_val:
            best_val=tmp_val
            best_softmax=softmax_tmp

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred=best_softmax.predict(X_test)
test_accuracy=np.mean((y_test_pred==y_test))
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))