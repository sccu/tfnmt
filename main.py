#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from dataset import DataSet
from seq2seq_model import Seq2SeqModel

FLAGS = tf.app.flags.FLAGS
LOG = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG)

tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")
tf.app.flags.DEFINE_integer("max_data_size", 100000, "Maximum data size")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("cell_size", 20, "LSTM cell size")
tf.app.flags.DEFINE_integer("seq_len", 10, "Maximum sequence length")
tf.app.flags.DEFINE_integer("stack_size", 2, "RNN stack size")
tf.app.flags.DEFINE_integer("embedding_size", 10, "Word embedding size")
tf.app.flags.DEFINE_integer("vocab_size", 50000, "Vocab size")
tf.app.flags.DEFINE_integer("learning_rate", 0.01, "Learning rate")



def main(argv=None):
  LOG.info("Preparing dataset...")
  data_manager = DataSet("train.zh", "train.kr", "test.zh", "test.kr", FLAGS.seq_len, FLAGS.vocab_size,
                         max_data_size=FLAGS.max_data_size)

  LOG.info("Building model...")
  model = Seq2SeqModel(FLAGS.cell_size, FLAGS.stack_size, FLAGS.batch_size, FLAGS.seq_len, FLAGS.vocab_size,
                       FLAGS.embedding_size, FLAGS.learning_rate)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in xrange(1, FLAGS.epochs + 1):
      for offset in range(0, 10000, FLAGS.batch_size):
        enc_inputs, dec_inputs = data_manager.get_batch(offset, FLAGS.batch_size)
        loss = model.step(sess, enc_inputs, dec_inputs)
        LOG.info("Epoch: %d, batch: %d, loss: %f", epoch, int(offset / FLAGS.batch_size) + 1, loss)


if __name__ == "__main__":
  tf.app.run()
