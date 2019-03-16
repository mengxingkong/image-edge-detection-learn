from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.metrics import confusion_matrix
from mnist import MNIST
import os


if __name__ == "__main__":
    # img_data = MNIST("/home/lijun/git/image-eage-detection-learn/data/MNIST/")
    # test_imgs,test_labels = img_data.load_testing()
    # train_imgs,train_labels = img_data.load_training()
    mnist = input_data.read_data_sets("/home/lijun/git/image-eage-detection-learn/data/MNIST/", one_hot=True)

    # Load data
    X_train = mnist.train.images
    Y_train = mnist.train.labels
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    # Get the next 64 images array and labels
    batch_X, batch_Y = mnist.train.next_batch(64)
