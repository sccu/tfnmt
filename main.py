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

tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size")
tf.app.flags.DEFINE_integer("max_data_size", 100000, "Maximum data size")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("cell_size", 500, "LSTM cell size")
tf.app.flags.DEFINE_integer("seq_len", 10, "Maximum sequence length")
tf.app.flags.DEFINE_integer("stack_size", 2, "RNN stack size")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Word embedding size")
tf.app.flags.DEFINE_integer("vocab_size", 50000, "Vocab size")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
tf.app.flags.DEFINE_integer("steps_per_print", 10, "Steps per print")


def main(argv=None):
  LOG.info("Preparing dataset...")
  data_manager = DataSet("train.zh", "train.kr", "test.zh", "test.kr", FLAGS.seq_len, FLAGS.vocab_size,
                         max_data_size=FLAGS.max_data_size)

  LOG.info("Building model...")
  model = Seq2SeqModel(FLAGS.cell_size, FLAGS.stack_size, FLAGS.batch_size, FLAGS.seq_len, FLAGS.vocab_size,
                       FLAGS.embedding_size, FLAGS.learning_rate)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("out")
    if ckpt and ckpt.model_checkpoint_path:
      LOG.info("Restoring a model from: %s", ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      LOG.info("Initializing a model...")
      sess.run(tf.global_variables_initializer())

    LOG.info("Writing graphs...")
    # tf.train.write_graph(sess.graph_def, "log", "train.pbtxt")

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
          total_loss = 0

          cv_offset = offset % (data_manager.get_testset_size() - FLAGS.batch_size)
          enc_inputs, dec_inputs = data_manager.get_test_batch(cv_offset, FLAGS.batch_size)
          cv_loss = model.step(sess, enc_inputs, dec_inputs, global_step, trainable=False)
          cv_ppl = np.exp(cv_loss)

          # cross-validation test and write checkpoint file.
          if (offset / FLAGS.batch_size + 1) % (20 * FLAGS.steps_per_print) == 0:
            save_path = saver.save(sess, "out/model.ckpt-%02d-%.3f" % (epoch, cv_ppl), global_step)
            LOG.info("Model saved in the file: %s", save_path)
            inferences = model.inference(sess, enc_inputs, dec_inputs)
            for i in range(5):
              LOG.debug("  source: [%s]", data_manager.src_ids_to_str(enc_inputs[i]))
              LOG.debug("  target: [%s]", data_manager.tgt_ids_to_str(dec_inputs[i]))
              LOG.debug("  inference: [%s]", data_manager.tgt_ids_to_str(inferences[i]))

    saver.save(sess, "out/model.ckpt-%02d" % epoch)


if __name__ == "__main__":
  tf.app.run()
