import numpy as np

"""
This file defines layer types that are commonly for recurrent neural
networks.
"""

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    #pass
    tmp = np.dot(x,Wx)    # it turn the dimension of x D to H,it means change x to tmp
    tmp1 = np.dot(prev_h,Wh)
    next_tmp = tmp+tmp1+b
    next_h = (np.exp(next_tmp) - np.exp(-next_tmp))/(np.exp(next_tmp)+np.exp(-next_tmp))
    cache = (next_tmp,Wx,x,prev_h,Wh,next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    next_tmp, Wx, x ,prev_h, Wh, next_h= cache

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    #pass
    dnext_tmp = dnext_h*(1 - np.power(next_h,2))
    db = np.sum(dnext_tmp,axis=0)
    dtmp = dnext_tmp
    dtmp1 = dnext_tmp
    dx = np.dot(dtmp, Wx.T)
    dWx = np.dot(x.T, dtmp)
    dWh = np.dot(prev_h.T, dtmp1)
    dprev_h = np.dot(dtmp,Wh.T)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db



def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    #pass
    N, T, D = x.shape
    H = b.shape[0]
    h = np.zeros([N, T, H])
    prev_h_one = h0
    input_x = np.zeros([N,D])
    next_tmp = np.zeros([N, T, H])
    next_h = np.zeros([N, T, H])
    prev_h = np.zeros([N, T, H])
    for i in range(T):
        input_x = x[:,i,:]
        next_one_h, cache_one = rnn_step_forward(x=input_x ,prev_h=prev_h_one, Wx=Wx,Wh=Wh, b=b)
        prev_h_one = next_one_h
        h[:,i,:] = next_one_h
        next_tmp[:, i, :] = cache_one[0]
        next_h[:, i, :] = cache_one[5]
        prev_h[:, i, :] = cache_one[3]

    cache = (next_tmp, Wx, x, prev_h, Wh, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache



def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    next_tmp, Wx, x, prev_h, Wh, next_h = cache
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    #pass
    N, T, D = x.shape
    H = Wh.shape[0]
    dx = np.zeros([N, T, D])
    dh0 = np.zeros([N, H])
    dWx = np.zeros([D, H])
    dWh = np.zeros([H, H])
    db = np.zeros([H,])
    dprev_h = np.zeros([N, H])
    rev_T = list(reversed(range(T)))
    for i in rev_T:
        dnext_h = dh[:,i,:] + dprev_h
        cache_tmp = (next_tmp[:, i, :], Wx, x[:, i, :], prev_h[:, i, :], Wh, next_h[:, i, :])
        dx[:, i, :], dprev_h , dWx_tmp, dWh_tmp, db_tmp = rnn_step_backward(dnext_h=dnext_h,cache=cache_tmp)
        dWx = dWx + dWx_tmp
        dWh = dWh + dWh_tmp
        db = db + db_tmp
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    #pass
    N , T = x.shape
    V , D = W.shape
    out = np.zeros([N , T , D])
    for i in range(N):
        for j in range(T):
            out[i,j,:] = W[x[i,j],:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    cache = (W, x)
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    #pass
    W, x = cache
    N , T = x.shape
    V , D  = W.shape
    dW = np.zeros([V , D])
    for i in range(V):
        for j in range(N):
            for z in range(T):
                if(x[j,z] == i):
                    dW[i,:] = dW[i,:] + dout[j,z,:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0 )
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def tanh(x):
    """
    compute tanh activation function based on x
    """
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return result

def d_sigmoid(x):
    """
    compute the derivatives of sigmoid function
    """
    temp = sigmoid(x)
    result = temp * (1 - temp)
    return result

def d_tanh(x):
    """
     compute the derivatives of tanh function
    """
    temp = tanh(x)
    result = 1 - np.power(temp,2)
    return result

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    #pass
    N, H = prev_h.shape
    _, D = x.shape
    next_h = np.zeros((N, H))
    next_c = np.zeros((N, H))
    gate_tmp = np.dot(x,Wx) + np.dot(prev_h, Wh) + b
    sigmoid_gate = sigmoid(gate_tmp[:,0:3*H])
    g_gate = tanh(gate_tmp[:,3*H:4*H])
    i_gate = sigmoid_gate[:,0:H]
    f_gate = sigmoid_gate[:,H:2*H]
    o_gate = sigmoid_gate[:,2*H:3*H]
    next_c = f_gate * prev_c + i_gate * g_gate
    next_h = o_gate * tanh(next_c)
    cache = (next_c, o_gate, f_gate, prev_c, g_gate, i_gate, gate_tmp, Wh, Wx, x, prev_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, next_c, cache



def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    dprev_h, dprev_c = None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    #pass
    next_c, o_gate, f_gate, prev_c, g_gate, i_gate, gate_tmp, Wh, Wx, x, prev_h = cache
    N, H = prev_c.shape
    do_gate = dnext_h * tanh(next_c)
    dnext_c = dnext_c + dnext_h * o_gate * d_tanh(next_c)
    dprev_c = dnext_c * f_gate
    df_gate = dnext_c * prev_c
    di_gate = dnext_c * g_gate
    dg_gate = dnext_c * i_gate
    dgate_tmp = np.zeros_like(gate_tmp)
    dgate_tmp[:,0:H] = d_sigmoid(gate_tmp[:,0:H]) * di_gate
    dgate_tmp[:,H:2*H] = d_sigmoid(gate_tmp[:,H:2*H]) * df_gate
    dgate_tmp[:,2*H:3*H] = d_sigmoid(gate_tmp[:,2*H:3*H]) * do_gate
    dgate_tmp[:,3*H:4*H] = d_tanh(gate_tmp[:,3*H:4*H]) * dg_gate
    dprev_h = np.dot(dgate_tmp,Wh.T)
    dWh =  np.dot(prev_h.T,dgate_tmp)
    dWx = np.dot(x.T, dgate_tmp)
    dx = np.dot(dgate_tmp, Wx.T)
    db = np.sum(dgate_tmp, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    #pass
    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros((N, T, H))
    cache_h = np.zeros((N, T+1, H))
    cache_h[:, 0, :] = h0
    prev_h = h0
    prev_c = np.zeros((N,  H))
    c = np.zeros((N, T+1, H))
    o_gate = np.zeros((N, T, H))
    f_gate = np.zeros((N, T, H))
    g_gate = np.zeros((N, T, H))
    i_gate = np.zeros((N, T, H))
    gate_tmp = np.zeros((N, T, 4*H))
    for i in range(T):
        next_h, next_c, cache_tmp = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:,i,:] = next_h
        cache_h[:, i+1, :] =  next_h
        c[:,i+1,:] = cache_tmp[0]
        o_gate[:, i, :] = cache_tmp[1]
        f_gate[:, i, :] = cache_tmp[2]
        g_gate[:, i, :] = cache_tmp[4]
        i_gate[:, i, :] = cache_tmp[5]
        gate_tmp[:, i, :] = cache_tmp[6]

    cache = (c, o_gate, f_gate, g_gate, i_gate, gate_tmp, Wh, Wx, x, cache_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    #pass
    c, o_gate, f_gate, g_gate, i_gate, gate_tmp, Wh, Wx, x, h = cache
    N, T, D = x.shape
    H, _ = Wh.shape
    dx = np.zeros_like(x)
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H, ))
    rev_T = list(reversed(range(T)))
    dnext_h = np.zeros((N, H))
    dc = np.zeros((N, T, H))
    dprev_c = np.zeros((N, H))
    dprev_h = np.zeros((N, H))
    dnext_c = np.zeros((N, H))
    for i in rev_T:
        dnext_h = dh[:, i, :] + dprev_h
        dnext_c = dprev_c + dc[:, i, :]
        cache_one = c[:, i+1, :], o_gate[:, i, :], f_gate[:, i, :], c[:, i, :], g_gate[:, i, :], i_gate[:, i, :], gate_tmp[:, i, :], Wh, Wx, x[:, i, :], h[:, i, :]
        dx_one, dprev_h_one, dprev_c_one, dWx_one, dWh_one, db_one = lstm_step_backward(dnext_h=dnext_h, dnext_c=dnext_c, cache=cache_one)
        dx[:, i, :] = dx_one
        dprev_h = dprev_h_one
        dprev_c = dprev_c_one
        dh0 = dprev_h_one
        dWx = dWx + dWx_one
        dWh = dWh + dWh_one
        db = db + db_one
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db





def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N , T , D = x.shape
    _, M = w.shape
    out = np.zeros([N,T,M])
    cache = None
    for i in range(N):
        for j in range(T):
            out[i,j,:] = np.dot(x[i,j,:],w)+b
    cache = (x , w)
    return out,cache



def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x , w = cache
    N, T, M =  dout.shape
    _, _, D = x.shape
    db = np.sum(dout,axis=(0,1))
    x_tmp = np.reshape(x, newshape=(N*T, D))
    dout_tmp = np.reshape(dout, newshape=(N*T, M))
    dw = np.dot(x_tmp.T,dout_tmp)
    dx_tmp = np.dot(dout_tmp,w.T)
    dx = np.reshape(dx_tmp,newshape=(N,T,D))
    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    N, T, V = x.shape
    loss = 0
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(T):
            if mask[i,j] == True:
                loss_tmp = -np.log(np.exp(x[i,j,y[i,j]]-np.max(x[i,j,:]))/np.sum(np.exp(x[i,j,:]-np.max(x[i,j,:]))))
                loss = loss + loss_tmp
                dx[i,j,:] = np.exp(x[i,j,:]-np.max(x[i,j,:]))/np.sum(np.exp(x[i,j,:]-np.max(x[i,j,:])))
                dx[i,j,y[i,j]] = -1 + np.exp(x[i,j,y[i,j]]-np.max(x[i,j,:]))/np.sum(np.exp(x[i,j,:]-np.max(x[i,j,:])))
    loss = loss/N
    dx = dx/N
    return loss, dx