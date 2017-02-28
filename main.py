#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import tensorflow as tf

from dataset import DataSet
from seq2seq_model import Seq2SeqModel

FLAGS = tf.app.flags.FLAGS
LOG = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    level=logging.DEBUG)

tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_integer("max_data_size", 1000000, "Maximum data size")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("cell_size", 600, "LSTM cell size")
tf.app.flags.DEFINE_integer("seq_len", 30, "Maximum sequence length")
tf.app.flags.DEFINE_integer("stack_size", 2, "RNN stack size")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Word embedding size")
tf.app.flags.DEFINE_integer("vocab_size", 50000, "Vocab size")
tf.app.flags.DEFINE_float("learning_rate", 1.0, "Learning rate")
tf.app.flags.DEFINE_integer("steps_per_print", 10, "Steps per print")
tf.app.flags.DEFINE_integer("steps_per_save", 200, "Steps per save")
tf.app.flags.DEFINE_float("dropout", 0.3, "Dropout")


def main(argv=None):
  LOG.info("Preparing dataset...")
  data_manager = DataSet("train.zh", "train.kr", "test.zh", "test.kr",
                         FLAGS.seq_len, FLAGS.vocab_size,
                         max_data_size=FLAGS.max_data_size)

  with tf.Session() as sess:
    LOG.info("Building model...")
    model = Seq2SeqModel(sess, FLAGS.cell_size, FLAGS.stack_size,
                         FLAGS.batch_size, FLAGS.seq_len, FLAGS.vocab_size,
                         FLAGS.embedding_size, FLAGS.learning_rate,
                         dropout=FLAGS.dropout)
    saver = tf.train.Saver()

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
    losses = []
    cv_losses = []
    cv_ppl_history = []
    global_step = 0
    for epoch in xrange(1, FLAGS.epochs + 1):
      for offset in range(0,
                          data_manager.get_trainset_size() - FLAGS.batch_size,
                          FLAGS.batch_size):
        global_step += 1
        enc_inputs, dec_inputs = data_manager.get_batch(offset,
                                                        FLAGS.batch_size)
        loss = model.step(sess, enc_inputs, dec_inputs, global_step)
        losses.append(loss)
        if (offset / FLAGS.batch_size + 1) % FLAGS.steps_per_print == 0:
          avg_loss = np.average(losses)
          ppl = np.exp(min(10, avg_loss))
          losses = []
          LOG.info("Epoch: %d, batch: %d/%d, PPL: %.3f, Loss: %.3f, LR: %.3f",
                   epoch, int(offset / FLAGS.batch_size) + 1,
                   data_manager.get_trainset_size() / FLAGS.batch_size,
                   ppl, avg_loss, model.learning_rate.eval())

          cv_offset = offset % (
            data_manager.get_testset_size() - FLAGS.batch_size)
          enc_inputs, dec_inputs = data_manager.get_test_batch(cv_offset,
                                                               FLAGS.batch_size)
          cv_loss = model.step(sess, enc_inputs, dec_inputs, global_step,
                               trainable=False)
          cv_losses.append(cv_loss)

        # cross-validation test and write checkpoint file.
        if (offset / FLAGS.batch_size + 1) % FLAGS.steps_per_save == 0:
          cv_ppl = np.exp(np.average(cv_losses))
          cv_losses = []
          save_path = saver.save(sess,
                                 "out/model.ckpt-%02d-%.3f" % (epoch, cv_ppl),
                                 global_step)
          LOG.info("Model saved in the file: %s", save_path)
          translations = model.predict(sess, enc_inputs, dec_inputs).tolist()
          for i in range(min(5, FLAGS.batch_size)):
            LOG.debug("  source: [%s]",
                      data_manager.src_ids_to_str(enc_inputs[i]))
            LOG.debug("  target: [%s]",
                      data_manager.tgt_ids_to_str(dec_inputs[i]))
            LOG.debug("  translation: [%s]",
                      data_manager.tgt_ids_to_str(translations[i]))

          # decaying learning rate
          if len(cv_ppl_history) > 2 and cv_ppl > max(cv_ppl_history[-3:]):
            sess.run(model.learning_rate_decaying_op)
          cv_ppl_history.append(cv_ppl)

    saver.save(sess, "out/model.ckpt-%02d" % epoch)


if __name__ == "__main__":
  tf.app.run()
