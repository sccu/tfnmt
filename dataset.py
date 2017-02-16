import cPickle
import os
from collections import Counter

import logging

import sys
from google import protobuf

LOG = logging.getLogger()


class DataSet(object):
  BOS_ID = 0
  EOS_ID = 1
  PAD_ID = 2
  UNK_ID = 3
  BOS = "<BOS>"
  EOS = "<EOS>"
  PAD = "<PAD>"
  UNK = "<UNK>"
  predefined_words = [BOS, EOS, PAD, UNK]

  def get_vocab(self, corpus_path, out_dir):
    filename = os.path.basename(corpus_path)
    vocab_path = os.path.join(out_dir, filename + ".vocab")
    if os.path.exists(vocab_path):
      vocab = self.load_vocab(vocab_path)
    else:
      vocab = self.create_vocab(corpus_path)
      self.store_vocab(vocab, vocab_path)
    LOG.info("Vocab count of %s: %d", vocab_path, len(vocab))
    return vocab

  @classmethod
  def load_vocab(cls, vocab_path):
    LOG.info("Loading vocab: " + vocab_path)
    vocab = dict()
    for l in open(vocab_path):
      cols = l.strip().decode("utf-8").split("\t")
      w = cols[0]
      i = int(cols[1])
      vocab.update({w: i})
    return vocab

  @staticmethod
  def store_vocab(vocab, path):
    LOG.info("Storing vocab into: " + path)
    with open(path, "wt") as f:
      for w, i in vocab.iteritems():
        f.write("{}\t{}\n".format(w.encode("utf-8"), i))

  def create_vocab(self, corpus_path):
    LOG.info("Creating vocab from: " + corpus_path)
    vocab_counter = Counter()
    with open(corpus_path) as f:
      for i, line in enumerate(f):
        if (i + 1) % 100000 == 0:
          LOG.debug("\tProcessing %d lines for creating vocabulary...", i + 1)
        words = line.strip().decode("utf-8").split(" ")
        for w in words:
          vocab_counter[w] += 1
    words = self.predefined_words + [entry[0] for entry in
                                     vocab_counter.most_common(self.vocab_size - len(self.predefined_words))]
    return {w: i for i, w in enumerate(words)}

  def __init__(self, src_train, tgt_train, src_test, tgt_test, seq_len, vocab_size, max_data_size=sys.maxsize):
    self.seq_len = seq_len
    self.vocab_size = vocab_size

    root_dir = os.path.dirname(os.path.realpath(__file__))
    self.data_dir = os.path.join(root_dir, "data")
    self.out_dir = os.path.join(root_dir, "out")

    self.src_train_path = os.path.join(self.data_dir, src_train)
    self.tgt_train_path = os.path.join(self.data_dir, tgt_train)
    self.src_test_path = os.path.join(self.data_dir, src_test)
    self.tgt_test_path = os.path.join(self.data_dir, tgt_test)

    self.src_vocab = self.get_vocab(self.src_train_path, self.out_dir)
    self.tgt_vocab = self.get_vocab(self.tgt_train_path, self.out_dir)

    self.test_dataset = self.prepare_data(self.src_test_path, self.tgt_test_path, self.out_dir, max_data_size)
    self.train_dataset = self.prepare_data(self.src_train_path, self.tgt_train_path, self.out_dir, max_data_size)

  def prepare_data(self, src_corpus_path, tgt_corpus_path, out_dir, max_data_size):
    src_filename = os.path.basename(src_corpus_path)
    src_ids_path = os.path.join(out_dir, src_filename + ".ids")
    tgt_filename = os.path.basename(tgt_corpus_path)
    tgt_ids_path = os.path.join(out_dir, tgt_filename + ".ids")
    if os.path.exists(src_ids_path) and os.path.exists(tgt_ids_path):
      return self.restore_data(src_ids_path), self.restore_data(tgt_ids_path)
    else:
      LOG.info("Load data from: [%s, %s]", src_ids_path, tgt_ids_path)
      src_inputs, tgt_inputs = self.load_data(src_corpus_path, tgt_corpus_path, max_data_size)
      self.store_data(src_inputs, src_ids_path)
      self.store_data(tgt_inputs, tgt_ids_path)
      return src_inputs, tgt_inputs

  def load_data(self, src_path, tgt_path, max_data_size):
    src_inputs = []
    tgt_inputs = []
    with open(src_path) as src:
      with open(tgt_path) as tgt:
        for i, (s, t) in enumerate(zip(src, tgt)):
          if i >= max_data_size:
            break
          if (i + 1) % 100000 == 0:
            LOG.debug("\tLoading data %d lines...", i + 1)
          swords = s.strip().decode("utf-8").split()
          twords = t.strip().decode("utf-8").split()
          src_ids = [self.src_vocab.get(w, self.UNK_ID) for w in swords] + [self.EOS_ID]
          tgt_ids = [self.BOS_ID] + [self.tgt_vocab.get(w, self.UNK_ID) for w in twords] + [self.EOS_ID]
          if len(src_ids) >= self.seq_len or len(tgt_ids) >= self.seq_len:
            continue
          src_inputs.append(src_ids)
          tgt_inputs.append(tgt_ids)
    return src_inputs, tgt_inputs

  @staticmethod
  def store_data(ids_inputs, path):
    with open(path, "wt") as f:
      cPickle.dump(ids_inputs, f)

  @staticmethod
  def restore_data(ids_path):
    LOG.info("Restoring data from: %s", ids_path)
    with open(ids_path) as f:
      inputs = cPickle.load(f)
    return inputs

  def get_batch(self, offset, batch_size):
    src = self.train_dataset[0][offset:offset + batch_size]
    tgt = self.train_dataset[1][offset:offset + batch_size]

    enc_inputs = []
    dec_inputs = []
    for t in xrange(self.seq_len):
      enc_inputs.append([])
      dec_inputs.append([])
      enc_inputs[t]
      dec_inputs[t]
      for b in xrange(batch_size):
        enc_inputs[t].append(src[b][t] if t < len(src[b]) else self.PAD_ID)
        dec_inputs[t].append(tgt[b][t] if t < len(tgt[b]) else self.PAD_ID)

    return enc_inputs, dec_inputs

  def get_trainset_size(self):
    return len(self.train_dataset[0])
