import logging
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest

LOG = logging.getLogger()


def dot_score(ht, hs):
  return ht * hs


def attentional_hidden_state(ht, hiddens, score=dot_score):
  with tf.variable_scope("attn"):
    cell_size = ht.get_shape()[1].value
    attn_Wc = tf.get_variable("attn_Wc", [2 * cell_size, cell_size])
    attn_b = tf.get_variable("attn_b", [cell_size])
    # tf.summary.histogram('attn_Wc', attn_Wc)
    # tf.summary.histogram('attn_b', attn_b)

    scores = [score(ht, hs) for hs in hiddens]
    exps = [tf.exp(s) for s in scores]
    denom = tf.add_n(exps)
    aligns = [exp / denom for exp in exps]
    context = tf.add_n([a * hs for a, hs, in zip(aligns, hiddens)])
    concat = tf.concat([context, ht], axis=1)
    attentional_ht = tf.tanh(tf.matmul(concat, attn_Wc) + attn_b)
    return attentional_ht


class Seq2SeqModel(object):
  def __init__(self, cell_size, stack_size, batch_size, seq_len, vocab_size,
               embedding_size, learning_rate, learning_rate_decaying_factor=0.9,
               max_gradient_norm=5.0, dropout=0.3, BOS_ID=0, PAD_ID=2):
    self.BOS_ID = BOS_ID
    self.PAD_ID = PAD_ID
    self.cell_size = cell_size
    self.seq_len = seq_len
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.dropout_op = tf.placeholder(tf.float32)
    self.dropout = dropout
    self.global_step = tf.Variable(1, trainable=False)
    self.learning_rate = tf.Variable(learning_rate, trainable=False)
    self.learning_rate_decaying_op = self.learning_rate.assign(
      self.learning_rate * learning_rate_decaying_factor)

    num_samples = 512

    with tf.variable_scope("seq2seq"):
      w_t = tf.get_variable("proj_w", [vocab_size, cell_size])
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [vocab_size])
      self.output_projection = (w, b)

      tf.summary.histogram('output_proj_w', w)
      tf.summary.histogram('output_proj_b', b)

      self.for_inference = tf.placeholder(tf.bool)
      self.dec_placeholder = tf.placeholder(tf.int32, [batch_size, seq_len],
                                            "dec_inputs")
      self.enc_placeholder = tf.placeholder(tf.int32, [batch_size, seq_len],
                                            "enc_inputs")
      self.enc_inputs = []
      self.dec_inputs = []
      self.setup_input_placeholders(seq_len)
      self.dec_labels = self.dec_inputs[1:]

      LOG.info("Creating rnn encoder...")
      hiddens, enc_state = self.create_rnn_encoder(cell_size, stack_size,
                                                   batch_size)
      LOG.info("Creating rnn decoder...")
      ret = self.create_rnn_decoder(enc_state, cell_size, stack_size,
                                    hiddens=hiddens)
      self.dec_outputs = ret[:seq_len]
      self.inference_outputs = [tf.argmax(
        tf.matmul(o, self.output_projection[0]) + self.output_projection[1],
        axis=1) for o in self.dec_outputs]
      self.inference_outputs = tf.transpose(self.inference_outputs)

      # word id outputs

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(
          tf.nn.sampled_softmax_loss(local_w_t, local_b, labels, local_inputs,
                                     num_samples, vocab_size),
          dtype=tf.float32)

      self.loss = self.sequence_loss(self.dec_outputs, self.dec_labels,
                                     softmax_loss_function=sampled_loss)
      # Gradients and SGD update operation for training the model
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
      self.update_op = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                 global_step=self.global_step)

      # write logs
      tf.summary.scalar("PPL", tf.exp(self.loss))
      self.summary_op = tf.summary.merge_all()

  def setup_input_placeholders(self, seq_len):
    enc_inputs = tf.split(self.enc_placeholder, num_or_size_splits=seq_len,
                          axis=1)
    dec_inputs = tf.split(self.dec_placeholder, num_or_size_splits=seq_len,
                          axis=1)
    for i in xrange(seq_len):
      self.enc_inputs.append(tf.reshape(enc_inputs[i], [-1]))
      self.dec_inputs.append(tf.reshape(dec_inputs[i], [-1]))

  def sequence_loss(self, logits, labels, softmax_loss_function):
    with tf.variable_scope("loss") as scope:
      losses = []
      weights = []
      for logit, label in zip(logits, labels):
        weight = tf.cast(tf.not_equal(label, self.PAD_ID), tf.float32)
        cross_entropy = softmax_loss_function(logit, label)
        losses.append(cross_entropy * weight)
        weights.append(weight)
      log_ppl = tf.reduce_sum(tf.add_n(losses) / tf.reduce_sum(weights))
      return log_ppl

  def create_rnn_encoder(self, cell_size, stack_size, batch_size):
    with tf.variable_scope("rnn_encoder") as scope:
      cell = BasicLSTMCell(cell_size)
      if self.dropout != 0.0:
        cell = DropoutWrapper(cell, output_keep_prob=1 - self.dropout_op)
      cell = MultiRNNCell([cell] * stack_size)
      cell = EmbeddingWrapper(cell, self.vocab_size, self.embedding_size)

      # [batch_size, seq_len] => list of [batch_size, 1]
      state = cell.zero_state(batch_size, tf.float32)
      outputs = []
      for i in xrange(self.seq_len):
        if i > 0:
          scope.reuse_variables()
        # [batch_size, 1] => [batch_size]
        _, state = cell(self.enc_inputs[i], state)
        outputs.append(state)
      return outputs, state

  def create_rnn_decoder(self, encoder_state, cell_size, stack_size,
                         hiddens=None):
    """
    Make up an RNN decoder.

    :param encoder_state: enoder_state
    :param cell_size:
    :param stack_size:
    :param hiddens:
    :return: outputs is a list of tensors which shape is [seq_len, embedding_size]. state's shape is [None, cell_size]
    """
    with tf.variable_scope("rnn_decoder") as scope:
      cell = BasicLSTMCell(cell_size)
      if self.dropout != 0.0:
        cell = DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout_op)
      cell = MultiRNNCell([cell] * stack_size)
      cell = EmbeddingWrapper(cell, self.vocab_size, self.embedding_size)

      def feeder(for_inference=False):
        state = encoder_state
        outputs = []
        for i in xrange(self.seq_len):
          if i > 0:
            scope.reuse_variables()
          if for_inference and i > 0:
            next_input = tf.argmax(
              tf.matmul(emb_output, self.output_projection[0]) +
              self.output_projection[1], axis=1)
          else:
            next_input = tf.reshape(self.dec_inputs[i], [-1])
          emb_output, state = cell(next_input, state)
          if hiddens:
            hs = [h[-1][-1] for h in hiddens]
            output = attentional_hidden_state(emb_output, hs)
          else:
            output = emb_output
          outputs.append(output)
        state_list = nest.flatten(state)
        return outputs + state_list

      return control_flow_ops.cond(self.for_inference, lambda: feeder(True),
                                   lambda: feeder(False))

  def step(self, sess, enc_inputs, dec_inputs, trainable=True, writer=None):
    """

    :param sess:
    :param enc_inputs: [batch_size, seq_len] int32 array.
    :param dec_inputs: [batch_size, seq_len] int32 array.
    :param trainable:
    :return:
    """

    global_step = sess.run(self.global_step)
    feed_dict = {self.for_inference: False,
                 self.dropout_op: self.dropout if trainable else 0.0,
                 self.enc_placeholder: enc_inputs,
                 self.dec_placeholder: dec_inputs}
    output_list = [self.loss]
    if trainable:
      output_list.append(self.update_op)
    if writer:
      output_list.append(self.summary_op)

    results = sess.run(output_list, feed_dict=feed_dict)

    loss = results[output_list.index(self.loss)]
    if writer:
      summary = results[output_list.index(self.summary_op)]
      writer.add_summary(summary, global_step)
    return loss

  def predict(self, sess, enc_inputs, dec_inputs):
    feed_dict = {self.for_inference: True,
                 self.dropout_op: 0.0,
                 self.enc_placeholder: enc_inputs,
                 self.dec_placeholder: dec_inputs}
    return sess.run(self.inference_outputs, feed_dict)
