import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
from tensorflow.examples.tutorials.mnist import input_data
import os

tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("/home/lijun/git/image-eage-detection-learn/data/MNIST/", one_hot=True)

batch_size = 4096
num_classes = 10
num_features = 784
max_steps = 10000

learning_rate = 0.1

l1_regul = 0.
l2_reful = 1.
examples_per_layer = 1000
num_trees = 10
max_depth = 16

learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_reful / examples_per_layer

