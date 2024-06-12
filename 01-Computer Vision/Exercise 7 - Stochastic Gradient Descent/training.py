import tensorflow as tf

from dataset import get_datasets
from logistic import softmax, cross_entropy, accuracy, model
import logging
import argparse
import os


def get_module_logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def sgd(params, grad, lr, bs):
    """
    stochastic gradient descent implementation
    args:
    - params [list[tensor]]: model params
    - grad [list[tensor]]: param gradient such that params[0].shape == grad[0].shape
    - lr [float]: learning rate
    - bs [int]: batch_size
    """
    # IMPLEMENT THIS FUNCTION
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad / bs)



def training_loop(train_dataset, model, loss, optimizer):
    """
    training loop
    args:
    - train_dataset: 
    - model [func]: model function
    - loss [func]: loss function
    - optimizer [func]: optimizer func
    returns:
    - mean_loss [tensor]: mean training loss
    - mean_acc [tensor]: mean training accuracy
    """
    class_num = 43
    accuracies = []
    losses = []
    for X, Y in train_dataset:
        with tf.GradientTape() as tape:
            # IMPLEMENT THIS FUNCTION
            X /= 255
            y_hat = model(X)
            losses.aapend(tf.math.reduce_mean(tf.one_hot(Y, class_num)))
            grad_update = tape.gradient(loss, [W, b])
            sgd([W, b], grad_update, lr, X.shape[0])
            accuracies.append(accuracy(y_hat, Y))
            
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    mean_loss = tf.math.reduce_mean(losses)
    return mean_loss, mean_acc


def validation_loop(val_dataset, model):
    """
    training loop
    args:
    - train_dataset: 
    - model [func]: model function
    - loss [func]: loss function
    - optimizer [func]: optimizer func
    returns:
    - mean_acc [tensor]: mean validation accuracy
    """
    # IMPLEMENT THIS FUNCTION
    accuracies = []
    for X, Y in val_dataset:
        X /= 255
        y_hat = model(X)
        accuracies.append(accuracy(y_hat, Y))
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    return mean_acc


if __name__  == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--imdir', required=True, type=str,
                        help='data directory')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()    

    logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    # get the datasets
    imdir = os.path.join(DIR_SRC, 'GTSRB/Final_Training/Images')
    train_dataset, val_dataset = get_datasets(args.imdir)

    # set the variables
    num_inputs = 1024*3
    num_outputs = 43
    W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                    mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))

    # training! 
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch}')
        loss, acc = training_loop(train_dataset, model, 
                                    negative_log_likelihood, sgd)
        logger.info(f'Mean training loss: {loss}, mean training accuracy {acc}')
        acc = validation_loop(val_dataset, model)
        logger.info(f'Mean validation accuracy {acc}')
