import cPickle
import os
from collections import Counter

import logging
from google import protobuf

LOG = logging.getLogger()


class DataSet(object):
  EOS_ID = 0
  UNK_ID = 1
  EOS = "<EOS>"
  UNK = "<UNK>"

  @classmethod
  def get_vocab(cls, corpus_path, out_dir):
    filename = os.path.basename(corpus_path)
    vocab_path = os.path.join(out_dir, filename + ".vocab")
    if os.path.exists(vocab_path):
      vocab = cls.load_vocab(vocab_path)
    else:
      vocab = cls.create_vocab(corpus_path)
      cls.store_vocab(vocab, vocab_path)
    LOG.info("Vocab count of %s: %d", vocab_path, len(vocab))
    return vocab

  @classmethod
  def load_vocab(cls, vocab_path):
    LOG.info("Loading vocab: " + vocab_path)
    vocab = dict()
    for id, l in enumerate(open(vocab_path)):
      w = l.strip().decode("utf-8")
      vocab.update({w: id})
    return vocab

  @staticmethod
  def store_vocab(vocab, path):
    LOG.info("Storing vocab into: " + path)
    with open(path, "wt") as f:
      for w in vocab:
        f.write("{}\n".format(w.encode("utf-8")))

  @classmethod
  def create_vocab(cls, corpus_path):
    LOG.info("Creating vocab from: " + corpus_path)
    vocab_counter = Counter()
    with open(corpus_path) as f:
      for i, line in enumerate(f):
        if (i + 1) % 100000 == 0:
          print "Processing %d lines for creating vocabulary..." % (i + 1)
        words = line.strip().decode("utf-8").split(" ")
        for w in words:
          vocab_counter[w] += 1
    return [cls.EOS, cls.UNK] + vocab_counter.keys()

  def __init__(self):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    self.data_dir = os.path.join(root_dir, "data")
    self.out_dir = os.path.join(root_dir, "out")

    self.src_train_path = os.path.join(self.data_dir, "train.zh")
    self.tgt_train_path = os.path.join(self.data_dir, "train.kr")
    self.src_test_path = os.path.join(self.data_dir, "test.zh")
    self.tgt_test_path = os.path.join(self.data_dir, "test.kr")

    self.src_vocab = DataSet.get_vocab(self.src_train_path, self.out_dir)
    self.tgt_vocab = DataSet.get_vocab(self.tgt_train_path, self.out_dir)

    self.test_dataset = self.get_data(self.src_test_path, self.tgt_test_path, self.out_dir)
    self.train_dataset = self.get_data(self.src_train_path, self.tgt_train_path, self.out_dir)

  def get_data(self, src_corpus_path, tgt_corpus_path, out_dir):
    src_filename = os.path.basename(src_corpus_path)
    src_ids_path = os.path.join(out_dir, src_filename + ".ids")
    tgt_filename = os.path.basename(tgt_corpus_path)
    tgt_ids_path = os.path.join(out_dir, tgt_filename + ".ids")
    if os.path.exists(src_ids_path) and os.path.exists(tgt_ids_path):
      return self.restore_data(src_ids_path), self.restore_data(tgt_ids_path)
    else:
      LOG.info("Load data from: [%s, %s]", src_ids_path, tgt_ids_path)
      src_inputs, tgt_inputs = self.load_data(src_corpus_path, tgt_corpus_path)
      self.store_data(src_inputs, src_ids_path)
      self.store_data(tgt_inputs, tgt_ids_path)
      return src_inputs, tgt_inputs

  def load_data(self, src_path, tgt_path):
    src_inputs = []
    tgt_inputs = []
    with open(src_path) as src:
      with open(tgt_path) as tgt:
        for i, (s, t) in enumerate(zip(src, tgt)):
          if (i + 1) % 100000 == 0:
            print "Loading data %d lines..." % (i + 1)
          swords = s.strip().decode("utf-8").split()
          twords = t.strip().decode("utf-8").split()
          if len(swords) >= 80 or len(twords) >= 80:
            continue
          src_ids = [self.src_vocab.get(w, self.UNK_ID) for w in swords] + [self.EOS_ID]
          tgt_ids = [self.tgt_vocab.get(w, self.UNK_ID) for w in twords] + [self.EOS_ID]
          src_inputs.append(src_ids)
          tgt_inputs.append(tgt_ids)
    return src_inputs, tgt_inputs

  @staticmethod
  def store_data(ids_inputs, path):
    with open(path, "wt") as f:
      cPickle.dump(ids_inputs, f)
      protobuf

  @staticmethod
  def restore_data(ids_path):
    LOG.info("Restoring data from: %s", ids_path)
    with open(ids_path) as f:
      inputs = cPickle.load(f)
    return inputs
