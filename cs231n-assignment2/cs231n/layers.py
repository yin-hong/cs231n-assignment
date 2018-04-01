import numpy as np



def affine_forward(x,w,b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out=None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #reshape the input into rows
    x_res=np.reshape(x,(x.shape[0],-1))
    out=np.dot(x_res,w)+b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache=(x,w,b)
    return out,cache


def affine_backward(dout,cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x,w,b=cache
    dx,dw,db=None,None,None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_res=np.reshape(x,(x.shape[0],-1))
    dx=np.dot(dout,w.T)
    dw=np.dot(x_res.T,dout)
    db=np.sum(dout,axis=0)
    dx=dx.reshape(*x.shape)
    return dx,dw,db



def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out=None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out=np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache=x
    return out,cache


def relu_backward(dout,cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx,x=None,cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx=(x>0)*dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



def batchnorm_forward(x,gamma,beta,bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode=bn_param['mode']
    eps=bn_param.get('eps',1e-5)
    momentum=bn_param.get('momentum',0.9)
    N,D=x.shape
    running_mean=bn_param.get('running_mean',np.zeros(D,dtype=x.dtype))
    running_var=bn_param.get('running_var',np.zeros(D,dtype=x.dtype))
    out,cache=None,None
    if mode=='train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        #pass
        minibatch_mean=np.mean(x,axis=0)
        std_mean=np.power(x-minibatch_mean.reshape(1,-1),2)
        #minibatch_std=np.sum(std_mean,axis=0)/N
        minibatch_std=np.var(x,axis=0)
        x_hat=(x-minibatch_mean.reshape(1,-1))/np.sqrt(minibatch_std+eps).reshape(1,-1)
        out=gamma.reshape(1,-1)*x_hat+beta.reshape(1,-1)
        running_mean=momentum*running_mean+(1-momentum)*minibatch_mean
        running_var=momentum*running_var+(1-momentum)*minibatch_std
        cache=(x_hat,gamma,minibatch_std,eps,x,minibatch_mean)
        #cache=(gamma,x,minibatch_mean,minibatch_std,eps,x_hat)
        #cache=(x_hat,gamma,beta,minibatch_mean,minibatch_std,x,eps)
        #cache=(x,gamma,beta,x_hat,minibatch_mean,minibatch_std,eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    elif mode=='test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        #pass
        x_test_hat=(x-bn_param.get('running_mean').reshape(1,-1))/np.sqrt(bn_param.get('running_var')+eps).reshape(1,-1)
        out=gamma.reshape(1,-1)*x_test_hat+beta.reshape(1,-1)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"'%mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean']=running_mean
    bn_param['running_var']=running_var

    return out,cache



def batchnorm_backward(dout,cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx,dgamma,dbeta=None,None,None
    x_hat,gamma,minibatch_std,eps,x,minibatch_mean=cache
    N,D=x.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    #pass
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*x_hat,axis=0)
    dx_hat=dout*gamma
    dx_1=dx_hat*(1/np.sqrt(minibatch_std+eps))
    dx_hat_std=-0.5*np.sum(dx_hat*(x-minibatch_mean),axis=0)*np.power(minibatch_std+eps,-1.5)
    dmean_1=-np.sum(dx_hat*(1.0/(np.sqrt(minibatch_std+eps))),axis=0)
    dmean_2=-dx_hat_std*(np.sum(x-minibatch_mean,axis=0))*2/N
    dx_2=2*dx_hat_std*(x-minibatch_mean)/N
    dmean=dmean_1+dmean_2
    dx_3=dmean/N
    dx=dx_1+dx_2+dx_3
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout,cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx,dgamma,dbeta=None,None,None
    x_hat,gamma,minibatch_std,eps,x,minibatch_mean=cache
    N,D=x.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    #pass
    #dbeta=np.sum(dout,axis=0)
    #dgamma=np.sum(dout*x_hat,axis=0)
    #dx_hat_x=((1.0-1.0/N)*np.sqrt(minibatch_std+eps)-0.5*np.power(minibatch_std+eps,-0.5)*(1.0-(1.0/N))/N*(x-minibatch_mean))/(minibatch_std+eps)
    #dx=dout*gamma*dx_hat_x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

def dropout_forward(x,dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p,mode=dropout_param['p'],dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask=None
    out=None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        #pass
        mask=(np.random.rand(*x.shape)<p)/p
        out=x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    elif mode=='test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #pass
        out=x
        mask=None
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache=(dropout_param,mask)
    out=out.astype(x.dtype,copy=False)
    return out,cache


def dropout_backward(dout,cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param,mask=cache
    mode=dropout_param['mode']

    dx=None
    if mode=='train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        #pass
        dx=dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode=='test':
        dx=dout
    return dx

def conv_forward_naive(x,w,b,conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out=None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #pass
    pad_num=conv_param['pad']
    stride_num=conv_param['stride']
    x_pad=np.pad(x,((0,0),(0,0),(pad_num,pad_num),(pad_num,pad_num)),mode='constant',constant_values=0)
    N,C,H,W=x.shape
    #print(N,C,H,W)
    #print(x_pad.shape)
    F,w_c,HH,WW=w.shape
    out_H=1+(H+2*pad_num-HH)/stride_num
    out_W=1+(W+2*pad_num-WW)/stride_num
    out=np.zeros((N,F,out_H,out_W))
    for i in range(N):
        for j in range(F):
            H_start_position=0
            for a in range(out_H):
                W_start_position=0
                W_end_position=W_start_position+WW
                H_start_position=H_start_position+bool(a!=0)*stride_num
                H_end_position=H_start_position+HH
                for c in range(out_W):
                    out[i,j,a,c]=np.sum(x_pad[i,:,H_start_position:H_end_position,W_start_position:W_end_position]*w[j,:,:,:])+b[j]
                    #H_start_position=H_end_position+stride_num
                    #H_end_position=H_start_position+HH
                    W_start_position=W_start_position+stride_num
                    W_end_position=W_start_position+WW
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache=(x,w,b,conv_param)
    return out,cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #pass
    x,w,b,conv_param=cache
    stride_num=conv_param['stride']
    pad_num=conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_num, pad_num), (pad_num, pad_num)), mode='constant', constant_values=0)
    N,C,H,W=x.shape
    F,W_C,HH,WW=w.shape
    dx=np.zeros_like(x)
    dw=np.zeros_like(w)
    db=np.zeros_like(b)
    out_H=1+(H+2*pad_num-HH)/stride_num
    out_W=1+(W+2*pad_num-WW)/stride_num
    dx_pad=np.zeros_like(x_pad)
    for i in range(N):
        for j in range(F):
            H_start_position=0
            for a in range(out_H):
                W_start_position=0
                W_end_position=W_start_position+WW
                H_start_position=H_start_position+bool(a!=0)*stride_num
                H_end_position=H_start_position+HH
                for c in range(out_W):
                    db[j]=db[j]+dout[i,j,a,c]
                    x_mask=x_pad[i,:,H_start_position:H_end_position,W_start_position:W_end_position]
                    dx_pad[i,:,H_start_position:H_end_position,W_start_position:W_end_position]+=w[j,:,:,:]*dout[i,j,a,c]
                    dw[j,:,:,:]=dw[j,:,:,:]+x_mask*dout[i,j,a,c]
                    W_start_position = W_start_position + stride_num
                    W_end_position = W_start_position + WW
    dx=dx_pad[:,:,pad_num:-pad_num,pad_num:-pad_num]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def max_pool_forward_naive(x,pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    #pass
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    N,C,H,W=x.shape
    out_H=(H-pool_height)/stride+1
    out_W=(W-pool_width)/stride+1
    out=np.zeros((N,C,out_H,out_W))
    for i in range(N):
        for j in range(C):
            H_start_position=0
            for a in range(out_H):
                W_start_position=0
                W_end_position=W_start_position+pool_width
                H_start_position=H_start_position+bool(a!=0)*stride
                H_end_position=H_start_position+pool_height
                for b in range(out_W):
                    out[i,j,a,b]=np.max(x[i,j,H_start_position:H_end_position,W_start_position:W_end_position])
                    W_start_position=W_start_position+stride
                    W_end_position=W_start_position+pool_width
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout,cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    #pass
    x,pool_param=cache
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    N,C,H,W=x.shape
    dx=np.zeros_like(x)
    out_H=(H-pool_height)/stride+1
    out_W=(W-pool_width)/stride+1
    for i in range(N):
        for j in range(C):
            H_start_position=0
            for a in range(out_H):
                W_start_position=0
                W_end_position=W_start_position+pool_width
                H_start_position=H_start_position+bool(a!=0)*stride
                H_end_position=H_start_position+pool_height
                for b in range(out_W):
                    max_num=np.max(x[i, j, H_start_position:H_end_position, W_start_position:W_end_position])
                    max_num_index=np.where(max_num==x[i, j, H_start_position:H_end_position, W_start_position:W_end_position])
                    dx[i,j,max_num_index[0][0]+H_start_position,max_num_index[1][0]+W_start_position]=dout[i,j,a,b]
                    W_start_position=W_start_position+stride
                    W_end_position=W_start_position+pool_width
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #pass
    N,C,H,W=x.shape
    X_new=np.transpose(x,(0,3,2,1))
    X_newShape=np.reshape(X_new,(N*H*W,C))
    tmp_out,cache=batchnorm_forward(X_newShape,gamma,beta,bn_param)
    temp_out=np.reshape(tmp_out,(N,W,H,C))
    out=np.transpose(temp_out,(0,3,2,1))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    #pass
    N,C,H,W=dout.shape
    dx_tmp,dgamma,dbeta=batchnorm_backward(np.reshape(np.transpose(dout,(0,3,2,1)),(N*H*W,C)),cache)
    dx_temp=np.reshape(dx_tmp,(N,W,H,C))
    dx=np.transpose(dx_temp,(0,3,2,1))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x,y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N=x.shape[0]
    margins=x-x[range(N),list(y)].reshape(-1,1)+1
    margins[range(N),list(y)]=0
    loss_tmp=(margins>0)*margins
    loss=np.sum(loss_tmp)/N
    dx=np.zeros_like(x)
    dx[margins>0]=1
    dx[range(N),list(y)]=-np.sum(dx,axis=1)
    dx=dx/N
    return loss,dx


def softmax_loss(x,y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    num_trains=x.shape[0]
    x=x-np.max(x,axis=1).reshape(-1,1)
    tmp=np.log(np.sum(np.exp(x),axis=1))
    loss_tmp=-x[range(num_trains),list(y)]+tmp
    loss=np.sum(loss_tmp)/num_trains
    dx=np.zeros_like(x)
    softmax_output=np.exp(x)/np.sum(np.exp(x),axis=1).reshape(-1,1)
    dx=softmax_output
    dx[range(num_trains),list(y)]=dx[range(num_trains),list(y)]-1
    dx=dx/num_trains
    return loss,dx

























