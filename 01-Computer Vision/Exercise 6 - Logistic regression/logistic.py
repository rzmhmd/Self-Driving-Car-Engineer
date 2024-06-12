import tensorflow as tf
import numpy as np
from utils import check_softmax, check_acc, check_model, check_ce


def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    logits_numpy = logits.numpy()
    logits_numpy = np.exp(logits_numpy)
    soft_logits_np = logits_numpy / logits_numpy.sum()
    soft_logits = tf.convert_to_tensor(soft_logits_np)
    return soft_logits


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    # IMPLEMENT THIS FUNCTION
    masked_logits = tf.boolean_mask(scaled_logits, one_hot)
    nll = - tf.math.log(masked_logits)

    return nll


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


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # IMPLEMENT THIS FUNCTION
    argmax = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)
    acc = tf.math.reduce_sum(tf.cast(argmax == Y, tf.int32)) / Y.shape[0]
    return acc


if __name__ == '__main__':
    # checking the softmax implementation
    check_softmax(softmax)

    # checking the NLL implementation
    check_ce(cross_entropy)

    # check the model implementation
    check_model(model)

    # check the accuracy implementation
    check_acc(accuracy)
