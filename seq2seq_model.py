import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell


class Seq2SeqModel(object):
  def __init__(self, cell_size, stack_size, batch_size, seq_len, vocab_size, embedding_size, learning_rate):
    self.BOS_ID = 0
    self.seq_len = seq_len
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.learning_rate = learning_rate

    self.enc_inputs = []
    self.dec_inputs = []
    num_samples = 512

    with tf.variable_scope("seq2seq"):
      enc_outputs, enc_state = self.create_rnn_encoder(cell_size, stack_size, batch_size, seq_len, vocab_size,
                                                       embedding_size)
      dec_outputs, dec_state = self.create_rnn_decoder(enc_state, cell_size, stack_size, batch_size, seq_len,
                                                       vocab_size, embedding_size)
      self.dec_outputs = dec_outputs
      self.dec_labels = self.dec_inputs[1:]

      # If we use sampled softmax, we need an output projection.
      output_projection = None
      # Sampled softmax only makes sense if we sample less than vocabulary size.

      w_t = tf.get_variable("proj_w", [vocab_size, cell_size])
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(
          tf.nn.sampled_softmax_loss(local_w_t, local_b, labels, local_inputs, num_samples, vocab_size),
          dtype=tf.float32)

      self.loss = self.sequence_loss(dec_outputs, self.dec_labels, softmax_loss_function=sampled_loss)
      self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

      # write logs
      tf.summary.scalar("loss", self.loss)
      self.merged = tf.summary.merge_all()
      self.train_writer = tf.summary.FileWriter("log")

  def sequence_loss(self, logits, labels, softmax_loss_function):
    with tf.variable_scope("loss") as scope:
      losses = []
      for logit, label in zip(logits, labels):
        cross_entropy = softmax_loss_function(logit, label)
        losses.append(cross_entropy)
      log_ppl = tf.reduce_mean(tf.add_n(losses) / len(labels))
      return log_ppl

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
    :return: outputs is a list of tensors which shape is [seq_len, embedding_size]. state's shape is [None, cell_size]
    """
    with tf.variable_scope("rnn_decoder") as scope:
      lstm = core_rnn_cell.BasicLSTMCell(cell_size)
      stacked_lstm = core_rnn_cell.MultiRNNCell([lstm] * stack_size)
      embedded_cell = core_rnn_cell.EmbeddingWrapper(stacked_lstm, vocab_size, embedding_size)

      for i in xrange(seq_len):
        dec_input= tf.placeholder(tf.int32, [None], name="dec_input{}".format(i))
        self.dec_inputs.append(dec_input)

      state = encoder_state
      outputs = []
      for i in xrange(seq_len):
        if i > 0:
          scope.reuse_variables()
        output, state = embedded_cell(self.dec_inputs[i], state)
        outputs.append(output)
      return outputs, state

  def step(self, sess, enc_inputs, dec_inputs, global_step):
    """

    :param sess:
    :param enc_inputs: [seq_len, batch_size] int32 array.
    :param dec_inputs: [seq_len, batch_size] int32 array.
    :return:
    """
    feed_dict = {}
    for i in xrange(self.seq_len):
      feed_dict[self.enc_inputs[i].name] = enc_inputs[i]
      feed_dict[self.dec_inputs[i].name] = dec_inputs[i]

    if global_step % 100 == 0:
      loss, _, summary = sess.run([self.loss, self.optim, self.merged], feed_dict)
      self.train_writer.add_summary(summary, global_step)
      return loss
    else:
      return sess.run([self.loss, self.optim], feed_dict)[0]

