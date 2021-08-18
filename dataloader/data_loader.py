# /usr/bin/env python

import collections
import logging
import re

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.data import load
from sklearn.preprocessing import MinMaxScaler


def is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


class Tokenizer(object):
    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def vocab_map(self, vocab):
        vocab_dict = collections.defaultdict(int, {k: v for v,
                                                            k in
                                                   enumerate(
                                                       vocab)})
        return vocab_dict

    def reverse_vocab_map(self):
        self.reverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def numbers_to_text(self, numbers_array):
        out_text = "".join(
            [self.reverse_vocab_dict[x] for x in numbers_array if x != 1])
        return out_text

    def clean_text(self, text):
        """First remove html markup then remove any non utf-8 character set
        """
        return re.sub(r'[^\x00-\x7F]|\n|\t|\r', ' ',
                      BeautifulSoup(str(text)).text).lower().strip()

    def tokenize_text(self, text):
        """
        Convert a given text in to a list of integer/embedding tokens
        Args:
            text: a sentence text

        Returns:
            Integer list representation of a text
        """
        if self.lowercase:
            text = text.lower()

        pos_tags = nltk.pos_tag(text)
        if len(text) >= self.seq_length:
            text = text[:self.seq_length]
            pos_tags = pos_tags[:self.seq_length]
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tags = [p[1] for p in pos_tags]

        else:
            padding = self.seq_length - len(text)
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tokens.extend([1] * padding)
            text_tags = [p[1] for p in pos_tags]
            text_tags.extend([1] * padding)
        return [text_tokens, text_tags]


class WordTokenizer(Tokenizer):
    """Load embeddings from a file
    """

    def __init__(self,
                 embeddings_path="/artifacts/pretrained-embeddings/glove-6B" +
                                 "-300d/1.0.0/glove.6B.300d.txt",
                 sequence_length=100,
                 lowercase=True):
        self.glove_vec_dict = None
        self.seq_length = sequence_length
        self.vocab_dict = {}
        self.dim = None
        self.embedding_average = 0
        self.embedding_matrix = np.nan
        self.embeddings_path = embeddings_path
        self.max_vocab = 40000
        self.min_freq = 2
        self._load_glove_embeddings()
        super(WordTokenizer, self).__init__(lowercase=lowercase)

    def _load_glove_embeddings(self):
        embeddings_path = self.embeddings_path
        with open(embeddings_path, "rb") as f1:
            vector_file = f1.read()
        vector_file = vector_file.decode("utf-8")
        vector_file = vector_file.split("\n")
        vector_file.insert(0, "_pad_")
        vector_file.insert(0, "_unk_")
        # get all words that have an embedding in the embedding file
        self.vocab_list = [x.split()[0] for x in vector_file[:-1]]
        # map word to index
        self.vocab_dict = self.vocab_map(self.vocab_list)
        # Get the glove matrix. We start from index 2 because we added _unk_ and _pad_ above
        self.embedding_matrix = np.array(
            [x.split()[1:] for x in vector_file[2:-1]],
            dtype="float32")
        self.dim = self.embedding_matrix.shape[1]
        # we will set the _unk_ token to average of all vectors and pad to 0
        self.embedding_average = np.average(self.embedding_matrix,
                                            axis=0).reshape(1, -1)
        self.embedding_matrix = np.concatenate((self.embedding_average,
                                                np.zeros((1, self.dim)),
                                                self.embedding_matrix))

    def fit_vocab(self, text_list):
        X_tokens = []
        embedding_array = []
        response_vocab_list = []
        for txt in text_list:
            X_tokens.append(nltk.word_tokenize(self.clean_text(txt)))
        freq = collections.Counter(p for o in X_tokens for p in o)
        response_words = [o for o, c in freq.most_common(self.max_vocab) if
                          c > self.min_freq]
        for word in response_words:
            if (word in self.vocab_list) and (word not in response_vocab_list):
                response_vocab_list.append(word)
                embedding_array.append(
                    self.embedding_matrix[self.vocab_dict[word]])
        embedding_array = np.array(embedding_array, dtype="float32")
        response_vocab_list.insert(0, "_pad_")
        response_vocab_list.insert(0, "_unk_")
        self.vocab_list = response_vocab_list
        self.vocab_dict = collections.defaultdict(int, {v: k for k, v in
                                                        enumerate(
                                                            response_vocab_list)})
        # update embedding average
        self.embedding_average = np.average(embedding_array, axis=0).reshape(1,
                                                                             -1)
        self.embedding_matrix = np.concatenate(
            (self.embedding_average, np.zeros((1, self.dim)), embedding_array))


class WordPOSTokenizer(Tokenizer):
    """Load embeddings from a file
    """

    def __init__(self,
                 embeddings_path="/artifacts/pretrained-embeddings/glove-6B" +
                                 "-300d/1.0.0/glove.6B.300d.txt",
                 sequence_length=100,
                 embedding_size_words=50):
        self.glove_vec_dict = None
        self.seq_length = sequence_length
        self.embedding_size_words = embedding_size_words
        self.vocab_dict = {}
        self.dim = None
        self.embedding_average = 0
        self.embedding_matrix = np.nan
        self.embeddings_path = embeddings_path
        self.max_vocab = 40000
        self.min_freq = 2
        self.tagdict = load('help/tagsets/upenn_tagset.pickle')
        self._load_glove_embeddings()
        self._load_pos_tags()

    def _load_pos_tags(self):
        self.pos_tags_list = list(self.tagdict.keys())
        self.pos_tags_list.insert(0, "_pad_")
        self.pos_tags_list.insert(0, "_unk_")
        self.pos_tags_dict = self.vocab_map(self.pos_tags_list)

    def _load_glove_embeddings(self):
        embeddings_path = self.embeddings_path
        with open(embeddings_path, "rb") as f1:
            vector_file = f1.read()
        vector_file = vector_file.decode("utf-8")
        vector_file = vector_file.split("\n")
        vector_file.insert(0, "_pad_")
        vector_file.insert(0, "_unk_")
        # get all words that have an embedding in the embedding file
        self.vocab_list = [x.split()[0] for x in vector_file[:-1]]
        # map word to index
        self.vocab_dict = self.vocab_map(self.vocab_list)
        # Get the glove matrix. We start from index 2 because we added _unk_ and _pad_ above
        self.embedding_matrix = np.array(
            [x.split()[1:] for x in vector_file[2:-1]],
            dtype="float32")
        self.dim = self.embedding_matrix.shape[1]
        # we will set the _unk_ token to average of all vectors and pad to 0
        self.embedding_average = np.average(self.embedding_matrix,
                                            axis=0).reshape(1, -1)
        self.embedding_matrix = np.concatenate((self.embedding_average,
                                                np.zeros((1, self.dim)),
                                                self.embedding_matrix))

    def fit_vocab(self, text_list):
        X_tokens = []
        embedding_array = []
        response_vocab_list = []
        for txt in text_list:
            X_tokens.append(nltk.word_tokenize(self.clean_text(txt)))
        freq = collections.Counter(p for o in X_tokens for p in o)
        response_words = [o for o, c in freq.most_common(self.max_vocab) if
                          c > self.min_freq]
        for word in response_words:
            if (word in self.vocab_list) and (word not in response_vocab_list):
                response_vocab_list.append(word)
                embedding_array.append(
                    self.embedding_matrix[self.vocab_dict[word]])
        embedding_array = np.array(embedding_array, dtype="float32")
        response_vocab_list.insert(0, "_pad_")
        response_vocab_list.insert(0, "_unk_")
        self.vocab_list = response_vocab_list
        self.vocab_dict = collections.defaultdict(int, {v: k for k, v in
                                                        enumerate(
                                                            response_vocab_list)})
        # update embedding average
        self.embedding_average = np.average(embedding_array, axis=0).reshape(1,
                                                                             -1)
        self.embedding_matrix = np.concatenate(
            (self.embedding_average, np.zeros((1, self.dim)), embedding_array))

    def tokenize_text(self, text):
        """
        Convert a given text in to a list of integer/embedding tokens
        Args:
            text: a sentence text

        Returns:
            Integer list representation of a text
        """
        text = self.clean_text(text)
        text = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(text)
        if len(text) >= self.seq_length:
            text = text[:self.seq_length]
            pos_tags = pos_tags[:self.seq_length]
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tags = [self.pos_tags_dict[p[1]] for p in pos_tags]
        else:
            padding = self.seq_length - len(text)
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tokens.extend([0] * padding)
            text_tags = [self.pos_tags_dict[p[1]] for p in pos_tags]
            text_tags.extend([0] * padding)
        return [text_tokens, text_tags]


def tokenize_text_scores(token_model,
                         onehot,
                         train_text_array,
                         test_text_array=None,
                         test_set_included=True):
    x_train = []
    x_train_tags = []
    x_test = []
    x_test_tags = []

    for idx, txt in enumerate(train_text_array):
        word_tokens, tag_tokens = token_model.tokenize_text(txt)
        x_train.append(word_tokens)
        x_train_tags.append(tag_tokens)
        if idx % 5000 == 0:
            logging.info("Done with {} examples".format(idx))

    if test_set_included:
        for idx, txt in enumerate(test_text_array):
            word_tokens, tag_tokens = token_model.tokenize_text(txt)
            x_test.append(word_tokens)
            x_test_tags.append(tag_tokens)
            if idx % 50000 == 0:
                logging.info("Done with {} examples".format(idx))

    x_train_returned = []
    x_test_returned = []
    x_train_tags_returned = []
    x_test_tags_returned = []

    if onehot == 'deep-onehot':
        x_train_tags_onehot = []
        for response in x_train_tags:
            response_onehot = []
            for el in response:
                token_onehot = [0] * token_model.embedding_size_words
                token_onehot[el] = 1
                response_onehot.append(token_onehot)
            x_train_tags_onehot.append(response_onehot)

        x_train_tags_returned = np.array(x_train_tags_onehot, dtype="int").reshape(-1, token_model.seq_length,
                                                                                   token_model.embedding_size_words, 1)

        x_train_words_onehot = []
        for response in x_train:
            response_onehot = []
            for el in response:
                token_onehot = [0] * token_model.embedding_size_words
                response_onehot.append(token_onehot)
            x_train_words_onehot.append(response_onehot)

        x_train_returned = np.array(x_train_words_onehot, dtype="int").reshape(-1, token_model.seq_length,
                                                                               token_model.embedding_size_words, 1)

        if test_set_included:
            x_test_tags_onehot = []
            for response in x_test_tags:
                response_onehot = []
                for el in response:
                    token_onehot = [0] * token_model.embedding_size_words
                    token_onehot[el] = 1
                    response_onehot.append(token_onehot)
                x_test_tags_onehot.append(response_onehot)

            x_test_tags_returned = np.array(x_test_tags_onehot, dtype="int").reshape(-1, token_model.seq_length,
                                                                                     token_model.embedding_size_words,
                                                                                     1)

            x_test_words_onehot = []
            for response in x_test:
                response_onehot = []
                for el in response:
                    token_onehot = [0] * token_model.embedding_size_words
                    response_onehot.append(token_onehot)
                x_test_words_onehot.append(response_onehot)

            x_test_returned = np.array(x_test_words_onehot, dtype="int").reshape(-1, token_model.seq_length,
                                                                                 token_model.embedding_size_words, 1)

    else:
        x_train_returned = np.array(x_train, dtype="int").reshape(-1, token_model.seq_length)
        x_train_tags_returned = np.array(x_train_tags).reshape(-1, token_model.seq_length)
        if test_set_included:
            x_test_returned = np.array(x_test, dtype="int").reshape(-1, token_model.seq_length)
            x_test_tags_returned = np.array(x_test_tags).reshape(-1, token_model.seq_length)

    return x_train_returned, \
           x_test_returned, \
           x_train_tags_returned, \
           x_test_tags_returned


def split_index(inputs, targets, batch_size):
    N = len(inputs)
    range_list = list(range(0, N, batch_size))
    range_list.extend([N])
    for i in range(len(range_list) - 1):
        sl = slice(range_list[i], range_list[i + 1])
        yield inputs[sl], targets[sl]


def split_index_two_inputs(inputs1, inputs2, targets, batch_size):
    N = len(inputs1)
    range_list = range(0, N, batch_size)
    range_list.extend([N])
    for i in range(len(range_list) - 1):
        sl = slice(range_list[i], range_list[i + 1])
        yield inputs1[sl], inputs2[sl], targets[sl]


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """Loads in two arrays and returns a batch generator
    """
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_two_inputs(inputs1, inputs2, targets, batchsize,
                                   shuffle=True):
    """Loads in two arrays and returns a batch generator
    """
    assert len(inputs1) == len(targets) == len(inputs2)
    indices = np.arange(len(inputs1))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt]


def load_asap_data(data_path,
                   set_no,
                   embedding_path,
                   seq_length,
                   embedding_size_words,
                   onehot,
                   train_all_sets):
    data = pd.read_csv(data_path)

    if train_all_sets:
        data_input = data
    else:
        data_input = data[data['essay_set'] == set_no]

    data_train = data_input[data_input['set'] == 'train']
    data_test = data_input[data_input['set'] == 'dev']

    normalizer = MinMaxScaler()

    x_train = data_train['essay'].values
    x_test = data_test['essay'].values

    if train_all_sets:
        y_train = []
        y_test = []
        for set_no in [1, 2, 3, 4, 5, 6, 7, 8]:
            y_train_set = data_train[data_train['essay_set'] == set_no]['domain1_score'].values
            y_test_set = data_test[data_test['essay_set'] == set_no]['domain1_score'].values
            y_train_set = normalizer.fit_transform(y_train_set.reshape(-1, 1))
            y_test_set = normalizer.transform(y_test_set.reshape(-1, 1))
            y_train.extend(y_train_set)
            y_test.extend(y_test_set)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
    else:
        y_train = data_train['domain1_score'].values
        y_train = normalizer.fit_transform(y_train.reshape(-1, 1))
        y_test = data_test['domain1_score'].values
        y_test = normalizer.transform(y_test.reshape(-1, 1))

    token_model = WordPOSTokenizer(embeddings_path=embedding_path,
                                   sequence_length=seq_length, embedding_size_words=embedding_size_words)

    x_train_words, x_test_words, x_train_tags, x_test_tags = tokenize_text_scores(
        token_model,
        onehot,
        x_train,
        x_test)

    if onehot == 'deep-onehot':
        x_train = np.concatenate((x_train_words, x_train_tags), axis=3)
        x_test = np.concatenate((x_test_words, x_test_tags), axis=3)

    else:
        x_train = np.concatenate((np.expand_dims(x_train_words, axis=2),
                                  np.expand_dims(x_train_tags, axis=2)), axis=2)
        x_test = np.concatenate((np.expand_dims(x_test_words, axis=2),
                                 np.expand_dims(x_test_tags, axis=2)), axis=2)
    random_indices = np.random.permutation(x_train.shape[0])
    logging.info("Permuting data")
    logging.info("Train Shapes")
    logging.info("-" * 50)
    logging.info((x_train.shape, y_train.shape))
    logging.info("Test shape")
    logging.info("-" * 50)
    logging.info((x_test.shape, y_test.shape))

    x_train = x_train[random_indices, :]
    y_train = y_train[random_indices, :]

    return x_train, x_test, y_train, y_test, normalizer, token_model


def load_holdout_set(data_path,
                     set_no,
                     normalizer,
                     token_model,
                     onehot):
    data = pd.read_csv(data_path)
    data_input = data[data['essay_set'] == set_no]
    data_test = data_input[data_input['set'] == 'test']

    x_test = data_test['essay'].values
    y_test = data_test['domain1_score'].values

    y_test = normalizer.transform(y_test.reshape(-1, 1))

    x_test_words, _, x_test_tags, _ = tokenize_text_scores(token_model,
                                                           onehot,
                                                           x_test,
                                                           test_set_included=False)

    if onehot == 'deep-onehot':
        x_test = np.concatenate((x_test_words, x_test_tags), axis=3)

    else:
        x_test = np.concatenate((np.expand_dims(x_test_words, axis=2),
                                 np.expand_dims(x_test_tags, axis=2)), axis=2)

    return x_test, y_test


if __name__ == "__main__":
    load_asap_data(data_path="../data/training_data.csv",
                   set_no=1,
                   embedding_path="../data/glove.6B.50d.txt",
                   seq_length=100,
                   embedding_size_words=50)
