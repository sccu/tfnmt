#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import tensorflow as tf

from dataset import DataSet
from seq2seq_model import Seq2SeqModel

FLAGS = tf.app.flags.FLAGS
LOG = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG)

tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")
tf.app.flags.DEFINE_integer("max_data_size", 200000, "Maximum data size")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("cell_size", 500, "LSTM cell size")
tf.app.flags.DEFINE_integer("seq_len", 20, "Maximum sequence length")
tf.app.flags.DEFINE_integer("stack_size", 2, "RNN stack size")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Word embedding size")
tf.app.flags.DEFINE_integer("vocab_size", 50000, "Vocab size")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_integer("steps_per_print", 100, "Steps per print")
tf.app.flags.DEFINE_string("save_path", None, "Save path")


def main(argv=None):
  LOG.info("Preparing dataset...")
  data_manager = DataSet("train.zh", "train.kr", "test.zh", "test.kr", FLAGS.seq_len, FLAGS.vocab_size,
                         max_data_size=FLAGS.max_data_size)

  LOG.info("Building model...")
  model = Seq2SeqModel(FLAGS.cell_size, FLAGS.stack_size, FLAGS.batch_size, FLAGS.seq_len, FLAGS.vocab_size,
                       FLAGS.embedding_size, FLAGS.learning_rate)
  saver = tf.train.Saver()

  save_path_prefix = "out/model.ckpt"
  save_path = FLAGS.save_path
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("out")
    if ckpt and ckpt.model_checkpoint_path:
      LOG.info("Restoring a model from: %s", ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      LOG.info("Initializing a model...")
      sess.run(tf.global_variables_initializer())

    LOG.info("Start training...")
    total_loss = 0
    global_step = 0
    for epoch in xrange(1, FLAGS.epochs + 1):
      for offset in range(0, data_manager.get_trainset_size() - FLAGS.batch_size, FLAGS.batch_size):
        global_step += 1
        enc_inputs, dec_inputs = data_manager.get_batch(offset, FLAGS.batch_size)
        loss = model.step(sess, enc_inputs, dec_inputs, global_step)
        total_loss += loss
        if (offset / FLAGS.batch_size + 1) % FLAGS.steps_per_print == 0:
          ppl = np.exp(total_loss / FLAGS.steps_per_print)
          LOG.info("Epoch: %d, batch: %d/%d, PPL: %f", epoch, int(offset / FLAGS.batch_size) + 1,
                   data_manager.get_trainset_size() / FLAGS.batch_size, ppl)
          total_loss = 0
        if (offset / FLAGS.batch_size + 1) % (100 * FLAGS.steps_per_print) == 0:
          save_path = saver.save(sess, "model.ckpt-%02d-%.3f" % (epoch, ppl))
          LOG.info("Model saved in the file: %s", save_path)
    saver.save(sess, "model.ckpt-%02d-%.3f" % (epoch, ppl))


if __name__ == "__main__":
  tf.app.run()
