#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

FLAGS = tf.app.flags.FLAGS
LOG = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")


def get_batch(batch_size):
  degrees = np.random.randint(0, 360, batch_size)
  xs = np.cos(degrees * np.pi / 180)
  ys = np.sin(degrees * np.pi / 180)
  coords = np.transpose([xs, ys])
  return degrees, coords


def embedding_main(argv=None):

  cell_size = 20
  batch_size = 8
  seq_len = 10
  stack_size = 2
  num_embeddings = 10
  epoch = 20000

  with tf.variable_scope("angular"):
    params = tf.get_variable("params", [360, 2], tf.float32)
    degrees = tf.placeholder(tf.int32, batch_size, "degrees")
    coords = tf.placeholder(tf.float32, [batch_size, 2], "coords")
    LOG.info("%s, %s", degrees, coords)

    embed = tf.nn.embedding_lookup(params, degrees)
    loss = tf.reduce_mean(tf.square(tf.subtract(embed, coords)))
    optim = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in xrange(epoch):
      xs, labels = get_batch(batch_size)
      _, cost, inferences = sess.run([optim, loss, embed], feed_dict={degrees.name: xs, coords.name: labels})
      print "{}:{}, {} => {} / {}".format(i, cost, xs[0], labels[0], inferences[0])


def lstm_main():
  cell_size = 20
  embedding_size = 5
  seq_len = 5
  batch_size = 8

  with tf.variable_scope("lstm_main") as scope:
    cell = core_rnn_cell.BasicLSTMCell(cell_size)
    cell = core_rnn_cell.MultiRNNCell([cell] * 2)
    cell = core_rnn_cell.EmbeddingWrapper(cell, 10, embedding_size)
    print "output_size: %s, state_size: %s" % (cell.output_size, cell.state_size)

    inputs = []
    for i in xrange(seq_len):
      input = tf.placeholder(dtype=tf.int32, shape=[None], name="input{}".format(i))
      inputs.append(input)

    state = cell.zero_state(batch_size, tf.float32)
    for i in xrange(seq_len):
      output, state = cell(inputs[i], state, scope=scope)
      if i == 0:
        scope.reuse_variables()


if __name__ == "__main__":
  # embedding_main()
  lstm_main()
