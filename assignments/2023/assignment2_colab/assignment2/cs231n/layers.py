from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N = x.shape[0]
    X_reshape = x.reshape((N,-1)) # (N,D)
    out = X_reshape @ w + b # (N,M)(100, 3072)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N=x.shape[0]
    db=np.sum(dout,axis=0) # (1,M)
    dx=(dout @ w.T).reshape(x.shape) # (N,D)
    dw=x.reshape((N,-1)).T @ dout # (N,M)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    out=x * (x>=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    dx = np.ones(x.shape) * (x>0) * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # 前向传递
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1)
    N = x.shape[0]
    exp_s_yi = exp_x[np.arange(N), y]
    l = exp_s_yi / sum_exp_x
    log_l = -np.log(l)
    loss = np.sum(log_l) / N

    # 反向传播
    grad_loss = 1.0 / N
    grad_l = grad_loss * -1 / l.reshape((-1,1))
    tmp = (-exp_s_yi / (sum_exp_x * sum_exp_x)).reshape(-1,1)
    C = x.shape[1]
    tmp = tmp @ np.ones((1,C))
    tmp[np.arange(N), y] = (sum_exp_x - exp_s_yi) / (sum_exp_x * sum_exp_x)
    grad_exp_x = grad_l * tmp
    dx = grad_exp_x * exp_x

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
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
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        u = np.mean(x, axis=0) # 计算批均值（Batch Mean）
        var = np.var(x, axis=0) # 方程
        std = np.sqrt(var + eps) # 计算批标准差（Batch Standard Deviation）
        x_hat = (x - u) / std # 标准化数据（Standardized Data）
        out = gamma * x_hat + beta # 缩放和平移（Scaling and Shifting）

        shape = bn_param.get('shape', (N, D))
        axis = bn_param.get('axis', 0)
        cache = x, u, var, std, x_hat, gamma, eps, shape, axis # 保存反向传播所需的变量

        running_mean = momentum * running_mean + (1 - momentum) * u # 更新运行均值（Running Mean）
        running_var = momentum * running_var + (1 - momentum) * var  # 更新运行方差（Running Variance）

        # cache = (x, gamma, beta, x_hat, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        x_hat = (x - running_mean) / np.sqrt(running_var + eps) # 标准化数据（Standardized Data）
        out = gamma * x_hat + beta # 缩放和平移（Scaling and Shifting）

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, u, var, std, x_hat, gamma, eps, (N, D), axis = cache
    dgamma = np.sum(dout * x_hat, axis=axis) # (N, D) * (N, D) -> (D)
    dbeta = np.sum(dout * 1, axis=axis)

    dx_hat = dout * gamma # (N, D)
    dvar = np.sum(dx_hat * (x - u) * -0.5 * (var + eps) ** -1.5, axis) # sum是为了把(N, D) -> (D),这样做是因为只涉及std u var 不涉及x，应当是(D)
    dx_u1 = dvar * 2 * (x - u) / N
    dx_u2 = dx_hat * 1 / std
    dx_u = dx_u1 + dx_u2
    du = np.sum(dx_u * -1, axis)
    dx1 = dx_u * 1
    dx2 = du * (np.ones_like(x) / N)
    dx = dx1 + dx2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, u, var, std, x_hat, gamma, eps, (N, D), axis = cache
    dgamma = np.sum(dout * x_hat, axis=0)  # 计算dgamma
    dbeta = np.sum(dout, axis=0)  # 计算dbeta

    dx_hat = dout * gamma  # 计算dx_hat
    dvar = np.sum(dx_hat * (x - u) * -0.5 * (var + eps) ** -1.5, axis)
    du = np.sum(dx_hat * -1 / std, axis=0) + dvar * np.sum(-2 * (x - u), axis=0) / N  # 计算du=-1/std + dvar*dvar/du
    dx = dx_hat * (1 / std) + dvar * (2 * (x - u) / N) - du * (- 1 / N)  # 计算dx=1/std + du*du/dx + dvar*dvar/dx

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
   	# 对输入数据求转置
    x = x.T
    gamma, beta = np.atleast_2d(gamma).T, np.atleast_2d(beta).T

    # 直接复用bn代码
    u = np.mean(x, axis=0) # 计算层均值（Layer Mean） (D, N) * (D, N) -> (N)
    var = np.var(x, axis=0) # 方差
    std = np.sqrt(var + eps) # 计算批标准差（Batch Standard Deviation）
    x_hat = (x - u) / std # 标准化数据（Standardized Data）
    out = gamma * x_hat + beta # 缩放和平移（Scaling and Shifting）

    # 转置回来
    out = out.T

    shape = ln_param.get('shape', x.shape)
    axis = ln_param.get('axis', 0)
    cache = x, u, var, std, x_hat, gamma, eps, shape, axis # 保存反向传播所需的变量

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, u, var, std, x_hat, gamma, eps, (D, N), axis = cache
    dout = dout.T
    dgamma = np.sum(dout * x_hat, axis=1) # (D, N) * (D, N) -> (D)
    dbeta = np.sum(dout * 1, axis=1)

    dx_hat = dout * gamma # (D, N)
    dvar = np.sum(dx_hat * (x - u) * -0.5 * (var + eps) ** -1.5, axis) # sum是为了把(D, N) -> (N),这样做是因为只涉及std u var 不涉及x，应当是(D)
    dx_u1 = dvar * 2 * (x - u) / N
    dx_u2 = dx_hat * 1 / std
    dx_u = dx_u1 + dx_u2
    du = np.sum(dx_u * -1, axis)
    dx1 = dx_u * 1
    dx2 = du * (np.ones_like(x) / N)
    dx = dx1 + dx2

    dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
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
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        mask = (np.random.rand(*x.shape)<p) / p
        out = x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    H_out = int((H + 2 * pad - HH) / stride + 1)
    W_out = int((W + 2 * pad - WW) / stride + 1)
    # print(f"pad:{pad},stride:{stride},H:{H},HH:{HH},H_out:{H_out}")
    # print(f"pad:{pad},stride:{stride},W:{W},WW:{WW},W_out:{W_out}")
    # 给x的上下左右填充上pad个0
    x_pad = np.pad(array=x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0)
    # 将卷积核w转换成F * (C * HH * WW)的矩阵 (便于使用矩阵乘法)
    w_row = w.reshape(F, -1) # 生成空白输出便于后续循环填充
    out = np.zeros((N, F, H_out, W_out))

    # 开始卷积
    for n in range(N):  # 遍历样本
        for f in range(F):  # 遍历卷积核
            for i in range(H_out):  # 遍历高
                for j in range(W_out):  # 遍历宽
                    # 获取当前卷积窗口
                    window = x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    # 将卷积窗口拉成一行
                    window_row = window.reshape(1, -1)
                    # 计算当前卷积窗口和卷积核的卷积结果
                    out[n, f, i, j] = np.sum(window_row * w_row[f, :]) + b[f]
      
	 # 将pad后的x存入cache (省的反向传播的时候在计算一次)
    x = x_pad


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # 获取一些需要用到的数据
    x, w, b, conv_param = cache
    N, C, H, W = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param["stride"]  # 步长
    pad = conv_param["pad"]  # 填充数量

    # 计算卷积后的高和宽
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)

    # 给dx,dw,db分配空间
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # print(dout.shape,x.shape)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # 获取当前卷积窗口
                    window = x[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    # out = x * w + b
                    # 计算db
                    db[f] += 1 * dout[n, f, i, j]
                    # 计算dw
                    dw[f] += window * dout[n, f, i, j]
                    # 计算dx
                    dx[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]

    # 去掉dx的pad
    dx = dx[:, :, pad:H - pad, pad:W - pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N,C,H,W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)
    # print(f"pad:{pad},stride:{stride},H:{H},HH:{HH},H_out:{H_out}")
    # print(f"pad:{pad},stride:{stride},W:{W},WW:{WW},W_out:{W_out}")
    # 将卷积核w转换成F * (C * HH * WW)的矩阵 (便于使用矩阵乘法)
    # w_row = w.reshape(F, -1) # 生成空白输出便于后续循环填充
    out = np.zeros((N, C, H_out, W_out))

    # 开始卷积
    for n in range(N):  # 遍历样本
        for c in range(C):  # 遍历卷积核
            for i in range(H_out):  # 遍历高
                for j in range(W_out):  # 遍历宽
                    # 获取当前卷积窗口
                    window = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                    # 将卷积窗口拉成一行
                    # window_row = window.reshape(1, -1)
                    # 计算当前卷积窗口和卷积核的卷积结果
                    out[n, c, i, j] = np.max(window)
      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # 获取一些需要用到的数据
    x, pool_param = cache
    N, C, H, W = x.shape  # N个样本，C个通道，H_input高，W_input宽
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)

    dx = np.zeros_like(x)
    

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # 获取当前卷积窗口
                    window = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                    maxindex = np.argmax(window)
                    # 计算dx
                    dx[n, c, i * stride + maxindex // pool_width, j * stride + maxindex % pool_width] += dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    x = np.moveaxis(x, 1, -1).reshape(-1, C)  # 将C通道放到最后，然后reshape成二维数组
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)  # 调用batchnorm_forward
    out = np.moveaxis(out.reshape(N, H, W, C), -1, 1)  # 将C通道放到第二维，然后reshape成四维数组

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

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
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N, C, H, W = dout.shape  # N个样本，C个通道，H高，W宽
    dout = np.moveaxis(dout, 1, -1).reshape(-1, C)  # 将C通道放到最后，然后reshape成二维数组
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)  # 调用batchnorm_forward
    dx = np.moveaxis(dx.reshape(N, H, W, C), -1, 1)  # 将C通道放到第二维，然后reshape成四维数组

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽

    # 将C通道分成G组，每组有C//G个通道
    x = x.reshape(N, G, C // G, H, W)  # reshape成五维数组
    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)  # 求均值
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True)  # 求方差
    x_norm = (x - x_mean) / np.sqrt(x_var + eps)  # 归一化

    x_norm = x_norm.reshape(N, C, H, W)  # reshape成四维数组
    out = gamma * x_norm + beta  # 伸缩平移

    cache = (x, x_norm, x_mean, x_var, gamma, beta, G, eps)  # 缓存变量

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, x_norm, x_mean, x_var, gamma, beta, G, eps = cache  # 从缓存中取出变量
    N, C, H, W = dout.shape  # N个样本，C个通道，H高，W宽

    # 计算dgamma和dbeta
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)  # 求dgamma
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # 求dbeta

    # 准备数据
    x = x.reshape(N, G, C // G, H, W)  # reshape成五维数组

    m = C // G * H * W
    dx_norm = (dout * gamma).reshape(N, G, C // G, H, W)
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * np.power((x_var + eps), -1.5), axis=(2, 3, 4), keepdims=True)
    dx_mean = np.sum(dx_norm * (-1) / np.sqrt(x_var + eps), axis=(2, 3, 4), keepdims=True) + dx_var * np.sum(-2 * (x - x_mean), axis=(2, 3, 4),
                                                                                                             keepdims=True) / m
    dx = dx_norm / np.sqrt(x_var + eps) + dx_var * 2 * (x - x_mean) / m + dx_mean / m
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
