import tensorflow as tf


def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits te|nsor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    exp = tf.exp(logits)
    denom = tf.math.reduce_sum(exp, 1, keepdims=True)
    return exp / denom


def cross_entropy(scaled_logits, one_hot):
    """
    CE implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy
    """
    masked_logits = tf.boolean_mask(scaled_logits, one_hot) 
    return -tf.math.log(masked_logits)


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # calculate argmax
    argmax = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)

    # calculate acc
    acc = tf.math.reduce_sum(tf.cast(argmax == Y, tf.int32)) / Y.shape[0]
    return acc


def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # IMPLEMENT THIS FUNCTION
    x_np = X.numpy()
    x_np = x_np.flatten()
    w_np = W.numpy()
    b_np = b.numpy()
    softmax_in = np.multiply(x_np, w_np) + b_np
    softmax_in_tf = tf.convert_to_tensor(softmax_in)
    return softmax(softmax_in_tf)
