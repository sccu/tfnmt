import cPickle
import os
from collections import Counter

import logging

import sys
from google import protobuf

LOG = logging.getLogger()


class DataSet(object):
  BOS = "<BOS>"
  EOS = "<EOS>"
  PAD = "<PAD>"
  UNK = "<UNK>"
  predefined_words = [BOS, EOS, PAD, UNK]
  BOS_ID = predefined_words.index(BOS)
  EOS_ID = predefined_words.index(EOS)
  PAD_ID = predefined_words.index(PAD)
  UNK_ID = predefined_words.index(UNK)

  def get_words(self, corpus_path, out_dir):
    filename = os.path.basename(corpus_path)
    dict_path = os.path.join(out_dir, filename + ".dict")
    if os.path.exists(dict_path):
      words = self.load_dict(dict_path)
    else:
      words = self.extract_words(corpus_path)
      self.store_words(words, dict_path)
    LOG.info("Word count of %s: %d", dict_path, len(words))
    return words

  @classmethod
  def load_dict(cls, dict_path):
    LOG.info("Loading dict: " + dict_path)
    words = []
    for l in open(dict_path):
      word = l.strip().decode("utf-8")
      words.append(word)
    return words

  @staticmethod
  def store_words(words, path):
    LOG.info("Storing words into: " + path)
    with open(path, "wt") as f:
      for w in words:
        f.write("{}\n".format(w.encode("utf-8")))

  def extract_words(self, corpus_path):
    LOG.info("Creating vocab from: " + corpus_path)
    vocab_counter = Counter()
    with open(corpus_path) as f:
      for i, line in enumerate(f):
        if (i + 1) % 100000 == 0:
          LOG.debug("  Processing %d lines for creating vocabulary...", i + 1)
        words = line.strip().decode("utf-8").split(" ")
        for w in words:
          vocab_counter[w] += 1
    words = self.predefined_words + [entry[0] for entry in
      vocab_counter.most_common( self.vocab_size - len(self.predefined_words))]
    return words

  def __init__(self, src_train, tgt_train, src_test, tgt_test, seq_len,
      vocab_size, max_data_size=sys.maxsize, data_dir="data",
      out_dir="out"):
    self.seq_len = seq_len
    self.vocab_size = vocab_size

    self.data_dir = data_dir
    self.out_dir = out_dir

    self.src_train_path = os.path.join(self.data_dir, src_train)
    self.tgt_train_path = os.path.join(self.data_dir, tgt_train)
    self.src_test_path = os.path.join(self.data_dir, src_test)
    self.tgt_test_path = os.path.join(self.data_dir, tgt_test)

    self.src_words = self.get_words(self.src_train_path, self.out_dir)
    self.src_vocab = {w: i for i, w in enumerate(self.src_words)}
    self.tgt_words = self.get_words(self.tgt_train_path, self.out_dir)
    self.tgt_vocab = {w: i for i, w in enumerate(self.tgt_words)}

    self.test_dataset = self.prepare_data(self.src_test_path,
      self.tgt_test_path, self.out_dir,
      max_data_size)
    self.train_dataset = self.prepare_data(self.src_train_path,
      self.tgt_train_path, self.out_dir,
      max_data_size)

  def prepare_data(self, src_corpus_path, tgt_corpus_path, out_dir,
      max_data_size):
    src_filename = os.path.basename(src_corpus_path)
    src_ids_path = os.path.join(out_dir, src_filename + ".ids")
    tgt_filename = os.path.basename(tgt_corpus_path)
    tgt_ids_path = os.path.join(out_dir, tgt_filename + ".ids")
    if os.path.exists(src_ids_path) and os.path.exists(tgt_ids_path):
      return self.restore_data(src_ids_path), self.restore_data(tgt_ids_path)
    else:
      LOG.info("Load data from: [%s, %s]", src_ids_path, tgt_ids_path)
      src_inputs, tgt_inputs = self.load_data(src_corpus_path, tgt_corpus_path,
        max_data_size)
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
            LOG.debug("  Loading data %d lines...", i + 1)
          swords = s.strip().decode("utf-8").split()
          twords = t.strip().decode("utf-8").split()
          src_ids = [self.src_vocab.get(w, self.UNK_ID) for w in swords] + [
            self.EOS_ID]
          tgt_ids = [self.BOS_ID] + [self.tgt_vocab.get(w, self.UNK_ID) for w in
            twords] + [self.EOS_ID]
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
    for i in xrange(batch_size):
      enc_inputs.append(src[i] + [self.PAD_ID] * (self.seq_len - len(src[i])))
      dec_inputs.append(tgt[i] + [self.PAD_ID] * (self.seq_len - len(tgt[i])))

    return enc_inputs, dec_inputs

  def get_trainset_size(self):
    return len(self.train_dataset[0])

  def get_cvset_size(self):
    return len(self.test_dataset[0])

  def src_ids_to_words(self, ids):
    return [self.src_words[id] for id in ids]

  def tgt_ids_to_words(self, ids):
    return [self.tgt_words[id] for id in ids]

  def src_ids_to_str(self, ids):
    try:
      eos_index = ids.index(self.EOS_ID)
      ids = ids[:eos_index]
    except ValueError:
      pass
    return " ".join([self.src_words[id] for id in ids])

  def tgt_ids_to_str(self, ids):
    try:
      eos_index = ids.index(self.EOS_ID)
      ids = ids[:eos_index]
    except ValueError:
      pass
    return " ".join([self.tgt_words[id] for id in ids if id != self.BOS_ID])

  def get_test_batch(self, offset, batch_size):
    src = self.test_dataset[0][offset:offset + batch_size]
    tgt = self.test_dataset[1][offset:offset + batch_size]

    enc_inputs = []
    dec_inputs = []
    for i in xrange(batch_size):
      enc_inputs.append(src[i] + [self.PAD_ID] * (self.seq_len - len(src[i])))
      dec_inputs.append(tgt[i] + [self.PAD_ID] * (self.seq_len - len(tgt[i])))

    return enc_inputs, dec_inputs
