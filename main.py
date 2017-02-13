#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from dataset import DataSet

FLAGS = tf.app.flags.FLAGS
LOG = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")


def loss_function(probabilities, target_words):
  pass


class Seq2SeqModel(object):
  def __init__(self, cell_size, stack_size, seq_len, num_embeddings):
    self.enc_inputs = []
    with tf.get_variable_scope("seq2seq"):
      encoder_state = self.create_rnn_encoder(cell_size, stack_size, seq_len, num_embeddings)
      outputs = self.create_rnn_decoder(encoder_state, cell_size, stack_size, seq_len, num_embeddings)

  def create_rnn_encoder(self, cell_size, stack_size, seq_len, vocab_size, num_embeddings):
    with tf.variable_scope("rnn_encoder"):
      lstm = core_rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=False)
      stacked_lstm = core_rnn_cell.MultiRNNCell([lstm] * stack_size, state_is_tuple=False)
      embedded_cell = core_rnn_cell.EmbeddingWrapper(stacked_lstm, vocab_size, num_embeddings)

      for i in xrange(seq_len):
        input = tf.placeholder([None], tf.int32, name="enc_input{}".format(i))
        self.enc_inputs.append(input)

      state = tf.zeros([None, lstm.state_size])
      for i in xrange(seq_len):
        output, state = embedded_cell(self.enc_inputs[:, i], state)
      return state

  def create_rnn_decoder(self, encoder_state, cell_size, stack_size, seq_len, vocab_size, num_embeddings):
    with tf.variable_scope("rnn_decoder"):
      lstm = core_rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=False)
      stacked_lstm = core_rnn_cell.MultiRNNCell([lstm] * stack_size, state_is_tuple=False)
      self.tgt_embeddings = tf.get_variable("tgt_embeddings", [vocab_size, num_embeddings], tf.float32)

      initial_state = state = encoder_state
      outputs = []
      for i in xrange(seq_len):
        output, state = stacked_lstm(self.dec_inputs[:, i], state)
        outputs.append(output)
      final_state = state
      return outputs


def main(argv=None):
  data_manager = DataSet()

  cell_size = 20
  batch_size = 8
  seq_len = 10
  stack_size = 2
  num_embeddings = 10

  model = Seq2SeqModel(cell_size, stack_size, seq_len, num_embeddings)



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
