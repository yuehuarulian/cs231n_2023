from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 前向传播
    # print(X.shape[0],X.shape[1],W.shape[1]) # N D 10
    y_pre = X @ W # N*C
    y_pre_exp = np.exp(y_pre)
    y_pre_sum = np.sum(y_pre_exp,axis=1) # N*1
    N = X.shape[0]
    yi = y_pre_exp[np.arange(N),y]
    y_i_sum = yi/y_pre_sum  # y1/y_sum
    y_loss = -np.log(y_i_sum)  # -log(y_i_sum)
    loss = np.mean(y_loss) + reg * np.sum(W * W)

    # 反向传播
    grad_loss = 1.0
    grad_y_loss = np.ones((N,1)) * grad_loss / N # N*1
    grad_y_i_sum = -1/y_i_sum.reshape(-1,1) * grad_y_loss # N * 1

    C = W.shape[1]
    temp = -yi.reshape(-1,1) @ np.ones((1,C)) # C = 10 D = 3072
    temp[np.arange(N),y] = y_pre_sum - yi
    # print((y_i_sum).shape,temp.shape)
    grad_y_pre_exp = (temp / (y_pre_sum * y_pre_sum).reshape(-1,1)) * grad_y_i_sum # N * C

    grad_y_pre = y_pre_exp * grad_y_pre_exp
    grad_w = X.T @ grad_y_pre  # D * C
    dW = grad_w + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 前向传播
    # print(X.shape[0],X.shape[1],W.shape[1]) # N D 10
    y_pre = X @ W # N*C
    y_pre_exp = np.exp(y_pre)
    y_pre_sum = np.sum(y_pre_exp,axis=1) # N*1
    N = X.shape[0]
    yi = y_pre_exp[np.arange(N),y]
    y_i_sum = yi/y_pre_sum  # y1/y_sum
    y_loss = -np.log(y_i_sum)  # -log(y_i_sum)
    loss = np.mean(y_loss) + reg * np.sum(W * W)

    # 反向传播
    grad_loss = 1.0
    grad_y_loss = np.ones((N,1)) * grad_loss / N # N*1
    grad_y_i_sum = -1/y_i_sum.reshape(-1,1) * grad_y_loss # N * 1

    C = W.shape[1]
    temp = -yi.reshape(-1,1) @ np.ones((1,C)) # C = 10 D = 3072
    temp[np.arange(N),y] = y_pre_sum - yi
    # print((y_i_sum).shape,temp.shape)
    grad_y_pre_exp = (temp / (y_pre_sum * y_pre_sum).reshape(-1,1)) * grad_y_i_sum # N * C

    grad_y_pre = y_pre_exp * grad_y_pre_exp
    grad_w = X.T @ grad_y_pre  # D * C
    dW = grad_w + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
