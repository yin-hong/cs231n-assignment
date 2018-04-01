import  matplotlib
matplotlib.use('Agg')
import data_utils
from k_nearest_neighbor import KNearestNeighbor
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(10.0,8.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'
#cifar10_dir='E://cs231n//cifar-10-batches-py'
#read cifar-10 data
cifar10_dir='/home/hongyin/file/cifar-10-batches-py'
X_train,y_train,X_test,y_test=data_utils.load_CIFAR10(cifar10_dir)
print('Training data shape: ',X_train.shape)
print('Training labels shape: ',y_train.shape)
print('Test data shape: ',X_test.shape)
print('Test labels shape: ',y_test.shape)
'''
#plot some images
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes=len(classes)
samples_per_class=7
for y,cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    for i,idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(np.uint8(X_train[idx]))
        plt.axis('off')
        if i==0:
            plt.title(cls)
#plt.show()
plt.savefig('/home/hongyin/file/cs231n-assignment1/picFaster.jpg')
'''
#Subsample the data for more efficient code execution
num_training=5000
mask=list(range(num_training))
X_train=X_train[mask]
y_train=y_train[mask]
num_test=500
mask=list(range(num_test))
X_test=X_test[mask]
y_test=y_test[mask]

#Reshape the image data into rows
X_train=np.reshape(X_train,(X_train.shape[0],-1))
X_test=np.reshape(X_test,(X_test.shape[0],-1))
print(X_train.shape,X_test.shape)
#Create a KNN classifier instance,k=1


classifier=KNearestNeighbor()
classifier.train(X_train,y_train)
"""
dists=classifier.compute_distance_two_loops(X_test)
print('dists.shape is')
print(dists.shape)
#plt.imshow(dists,interpolation='none')
#plt.savefig('/home/hongyin/file/cs231n-assignment1/picFaster.jpg')
y_test_pred=classifier.predict_labels(dists,k=1)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


#k=5
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
"""

"""
# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one=classifier.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
"""

"""
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference=np.linalg.norm(dists-dists_two,ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
"""

"""
# Let's compare how fast the implementations are
def time_function(f,*args):
    import time
    tic=time.time()
    f(*args)
    toc=time.time()
    return toc-tic

two_loop_time=time_function(classifier.compute_distance_two_loops,X_test)
print('Two loop version took %f seconds'%two_loop_time)

one_loop_time=time_function(classifier.compute_distances_one_loop,X_test)
print('One loop version took %f seconds'%one_loop_time)

no_loop_time=time_function(classifier.compute_distances_no_loops,X_test)
print('No loop version took %f seconds'%no_loop_time)
# you should see significantly faster performance with the fully vectorized implementation
"""


#Cross-Validation
num_folds=5
k_choices=[1,3,5,8,10,12,15,20,50,100]
X_train_folds=[]
y_train_folds=[]
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds=np.array_split(X_train,num_folds)
y_train_folds=np.array_split(y_train,num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies={}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################

classifier_cross=KNearestNeighbor()
for k in k_choices:
    accuracy_list=list()
    for i in range(num_folds):
        X_test_temp=np.array(X_train_folds[i])
        y_test_temp=np.array(y_train_folds[i])
        X_train_tmp = np.array(X_train_folds[:i] + X_train_folds[i + 1:])
        y_train_tmp = np.array(y_train_folds[:i] + y_train_folds[i + 1:])
        X_train_tmp = X_train_tmp.reshape(-1, X_train_tmp.shape[2])
        y_train_tmp = y_train_tmp.reshape(-1)
        classifier_cross.train(X_train_tmp,y_train_tmp)
        dists=classifier_cross.compute_distances_no_loops(X_test_temp)
        y_test_pred = classifier.predict_labels(dists, k)
        num_correct = np.sum(y_test_pred == y_test_temp)
        num_test=X_test_temp.shape[0]
        accuracy = float(num_correct) / num_test
        accuracy_list.append(accuracy)
    k_to_accuracies[k]=accuracy_list

best_k=-1
max_accuracy=-1
for k in k_choices:
    accuracies=k_to_accuracies[k]
    avg_accuracy=sum(accuracies)/len(accuracies)
    print(k,' : ',avg_accuracy)
    if avg_accuracy>max_accuracy:
        max_accuracy=avg_accuracy
        best_k=k

print(best_k)


"""
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy_item in k_to_accuracies[k]:
        print('k=%d,accuracy=%f'%(k,accuracy_item))


# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.savefig('/home/hongyin/file/cs231n-assignment1/picFaster.jpg')
"""



# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 50

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists= classifier.compute_distances_no_loops(X_test)
y_test_pred=classifier.predict_labels(dists,k=best_k)
# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
num_test=500
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

