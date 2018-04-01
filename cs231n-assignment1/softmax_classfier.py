import numpy as np



def softmax_loss_naive(W,X,y,reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

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
    #Initialize the loss and gradient to zero
    loss=0.0
    dW=np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_traines=X.shape[0]
    num_classes=W.shape[1]
    for i in range(num_traines):
        z=np.dot(X[i,:],W)
        shift_z=z-max(z)
        loss_i=-shift_z[y[i]]+np.log(np.sum(np.exp(shift_z)))
        loss=loss+loss_i
        softmax_out = np.exp(shift_z) / np.sum(np.exp(shift_z))
        for j in range(num_classes):
            if j==y[i]:
                dW[:,j]=dW[:,j]+(np.transpose(X[i,:])*(softmax_out[y[i]]-1))
            else:
                dW[:,j]=dW[:,j]+np.transpose(X[i,:])*softmax_out[j]
    loss=loss/num_traines
    loss=loss+0.5*reg*np.sum(W*W)
    dW=dW/num_traines
    dW=dW+reg*W
    return loss,dW

def softmax_loss_vectorized(W,X,y,reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss=0.0
    dW=np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_trains=X.shape[0]
    num_classes=W.shape[1]
    scores=np.dot(X,W)
    shifted_scores=scores-np.max(scores,axis=1).reshape(-1,1)
    loss_tmp=-shifted_scores[range(num_trains),list(y)]+np.log(np.sum(np.exp(shifted_scores),axis=1))
    loss=np.sum(loss_tmp)
    loss=loss/num_trains+reg*0.5*np.sum(W*W)
    softmax_output=np.exp(shifted_scores)/np.sum(np.exp(shifted_scores),axis=1).reshape(-1,1)
    softmax_output[range(num_trains),list(y)]=softmax_output[range(num_trains),list(y)]-1
    dW=np.dot(X.T,softmax_output)
    dW=dW/num_trains+reg*W
    return loss,dW