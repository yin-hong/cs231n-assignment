import numpy as np
class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X,k=1,num_loops=0):
        if num_loops==0:
            dists=self.compute_distance_no_loops(X)
        else:
            raise ValueError('Invalid value %d num_loops'%num_loops)
        return self.predict_labels(dists,k=k)

    def compute_distance_two_loops(self,X):
        """
            Compute the distance between each test point in X and each training point
            in self.X_train using a nested loop over both the training data and the
            test data.

            Inputs:
            - X: A numpy array of shape (num_test, D) containing test data.

            Returns:
            - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
              is the Euclidean distance between the ith test point and the jth training
              point.
         """
        num_train=self.X_train.shape[0]
        num_test=X.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j]=np.sqrt(np.sum(np.power(X[i,:]-self.X_train[j,:],2)))
        return dists


    def compute_distances_one_loop(self,X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_train=self.X_train.shape[0]
        num_test=X.shape[0]
        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            temp=X[i,:]-self.X_train
            temp_power=np.power(temp,2)
            temp_sum=np.sum(temp_power,axis=1)
            dists[i,:]=np.sqrt(temp_sum)
        return dists

    def compute_distances_no_loops(self,X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_train=self.X_train.shape[0]
        num_test=X.shape[0]
        dists=np.zeros((num_test,num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        X_power_2=np.power(X,2)
        train_power_2=np.power(self.X_train,2)
        test_sum=np.sum(X_power_2,axis=1)
        train_sum=np.sum(train_power_2,axis=1)
        train_sum_tile=np.tile(train_sum,(num_test,1))
        print("train_sum_shape_tile is ")
        print(train_sum_tile.shape)
        temp_sum=np.transpose([test_sum])+train_sum_tile
        dists_temp=temp_sum-2*np.dot(X,self.X_train.T)
        dists=np.sqrt(dists_temp)
        return dists

    def predict_labels(self,dists,k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test=dists.shape[0]
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            closest_y=[]
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            num_test_sort_indics=np.argsort(dists[i,:])
            closest_y=num_test_sort_indics[0:k].tolist()
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            vote=[0]*10
            for j in closest_y:
                vote[int(self.y_train[j])]=vote[int(self.y_train[j])]+1
            max_num_label=-1;
            max_num=-1
            for j,num in enumerate(vote):
                if num>max_num:
                    max_num_label=j
                    max_num=num
            y_pred[i]=max_num_label
        return y_pred