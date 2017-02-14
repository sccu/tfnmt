#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

FLAGS = tf.app.flags.FLAGS
LOG = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")


class Seq2SeqModel(object):
  def __init__(self, cell_size, stack_size, batch_size, seq_len, vocab_size, embedding_size):
    self.enc_inputs = []
    self.dec_inputs = []
    with tf.variable_scope("seq2seq"):
      enc_outputs, enc_state = self.create_rnn_encoder(cell_size, stack_size, batch_size, seq_len, vocab_size,
                                                       embedding_size)
      dec_outputs, dec_state = self.create_rnn_decoder(enc_state, cell_size, stack_size, batch_size, seq_len,
                                                       vocab_size, embedding_size)

  def create_rnn_encoder(self, cell_size, stack_size, batch_size, seq_len, vocab_size, embedding_size):
    with tf.variable_scope("rnn_encoder") as scope:
      lstm = core_rnn_cell.BasicLSTMCell(cell_size)
      stacked_lstm = core_rnn_cell.MultiRNNCell([lstm] * stack_size)
      embedded_cell = core_rnn_cell.EmbeddingWrapper(stacked_lstm, vocab_size, embedding_size)

      for i in xrange(seq_len):
        enc_input = tf.placeholder(tf.int32, [None], name="enc_input{}".format(i))
        self.enc_inputs.append(enc_input)

      state = embedded_cell.zero_state(batch_size, tf.float32)
      # state = tf.zeros([None, cell_size], tf.float32)
      outputs = []
      for i in xrange(seq_len):
        if i > 0:
          scope.reuse_variables()
        output, state = embedded_cell(self.enc_inputs[i], state)
        outputs.append(output)
      return outputs, state

  def create_rnn_decoder(self, encoder_state, cell_size, stack_size, batch_size, seq_len, vocab_size, embedding_size):
    """
    Make up an RNN decoder.

    :param encoder_state: enoder_state
    :param cell_size:
    :param stack_size:
    :param batch_size:
    :param seq_len:
    :param vocab_size:
    :param embedding_size: embedding size.
    :return: outputs is a list of tensors which shape is [seq_len, ]. state which shape is [None,
    """
    with tf.variable_scope("rnn_decoder") as scope:
      lstm = core_rnn_cell.BasicLSTMCell(cell_size)
      stacked_lstm = core_rnn_cell.MultiRNNCell([lstm] * stack_size)
      embedded_cell = core_rnn_cell.EmbeddingWrapper(stacked_lstm, vocab_size, embedding_size)

      for i in xrange(seq_len):
        dec_input = tf.placeholder(tf.int32, [None], name="dec_input{}".format(i))
        self.dec_inputs.append(dec_input)

      initial_state = state = encoder_state
      output = "<BOS>"
      outputs = []
      for i in xrange(seq_len):
        if i > 0:
          scope.reuse_variables()
        output, state = embedded_cell(self.dec_inputs[i], state)
        outputs.append(output)
      return outputs, state


def main(argv=None):
  # data_manager = DataSet()

  cell_size = 20
  batch_size = 8
  seq_len = 10
  stack_size = 2
  num_embeddings = 10
  vocab_size = 1000

  model = Seq2SeqModel(cell_size, stack_size, batch_size, seq_len, vocab_size, num_embeddings)

  return

  # Initial state of the LSTM memory.
  state = tf.zeros([batch_size, lstm.state_size])
  probabilities = []
  softmax_w = tf.get_variable("softmax_w", [None, seq_len], tf.float)
  softmax_b = tf.get_variable("softmax_b", [seq_len], tf.float)
  loss = 0.0
  for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)


"""
  def cond(index, q):
    return tf.less(index, 10)

  def body(index, q):
    index = tf.add(index, 1)
    # q.enque(index)
    return [index, q]

  i = tf.constant(0)
  q = tf.FIFOQueue(100, tf.int32)
  r = tf.while_loop(cond, body, (i, q))

  sess = tf.InteractiveSession()
  print sess.run(r)
  # print l
"""

if __name__ == "__main__":
  tf.app.run()
