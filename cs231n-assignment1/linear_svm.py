import numpy as np

def svm_loss_naive(W,X,y,reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    note delta = 1
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    num_train=X.shape[0]
    num_class=W.shape[1]
    dw=np.zeros(W.shape)
    loss=0.0
    score = np.dot(X, W)
    for i in range(num_train):
        dw_tmp=0
        for j in range(num_class):
            if j==y[i]:
                continue
            else:
                margin=score[i,j]-score[i,y[i]]+1
                if margin>0:
                    dw[:,j]=dw[:,j]+np.transpose(X[i])
                    loss=loss+margin
                    dw[:,y[i]]+=-X[i].T
    dw=dw/num_train
    loss=loss/num_train+0.5*reg*np.sum(W*W)
    dw=dw+reg*W
    return loss,dw


def svm_loss_vectorized(W,X,y,reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """

    loss=0.0
    dw=np.zeros(W.shape)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train=X.shape[0]
    num_class=W.shape[1]
    score=np.dot(X,W)
    All_True_Label_Score=score[range(num_train),y.tolist()].reshape(-1,1)
    L_tmp=score-All_True_Label_Score+1
    L_tmp[range(num_train),y.tolist()]=-1
    L_tmp_Large_than_zero=L_tmp>0
    L_tmp=L_tmp*L_tmp_Large_than_zero
    loss=np.sum(L_tmp)/num_train+0.5*reg*np.sum(W*W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    coff_mat=np.zeros((num_train,num_class))
    coff_mat[L_tmp>0]=1
    coff_mat[range(num_train),y.tolist()]=0
    sum_row=np.sum(coff_mat,axis=1)
    #sum_row=np.sum(L_tmp_Large_than_zero,axis=1)
    coff_mat[range(num_train),y.tolist()]=-sum_row
    dw=np.dot(X.T,coff_mat)
    dw=dw/num_train+reg*W


    return loss,dw






