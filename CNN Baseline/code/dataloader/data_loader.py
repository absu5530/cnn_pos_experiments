# /usr/bin/env python

import pandas as pd
import string
import collections
from sklearn.preprocessing import OneHotEncoder
import logging
import numpy as np
import re
from bs4 import BeautifulSoup
import os
import subprocess
import nltk
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import text

def is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


class Tokenizer(object):
    def vocab_map(self, vocab):
        vocab_dict = collections.defaultdict(int, {k: v for v,
                                                                   k in
                                                          enumerate(
                                                              vocab)})
        return vocab_dict

    def reverse_vocab_map(self):
        self.reverse_vocab_dict = {v: k for k, v in self.vocab_dict.items()}

    def numbers_to_text(self, numbers_array):
        out_text = "".join([self.reverse_vocab_dict[x] for x in numbers_array if x != 1])
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
        text = self.clean_text(text)
        text = nltk.word_tokenize(text)
        if len(text) >= self.seq_length:
            text = text[:self.seq_length]
            text_tokens = [self.vocab_dict[t] for t in text]
        else:
            padding = self.seq_length - len(text)
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tokens.extend([1] * padding)
        return text_tokens


class TextTokenizer(Tokenizer):
    def __init__(self,
                 charecter_length=1024):
        self.vocab_dict = None
        self.vocab = string.punctuation + \
                     "abcdefghijklmnopqrstuvwxyz0123456789" + " "
        self.vocab = list(self.vocab)
        self.vocab.insert(0, "_pad_")
        self.vocab.insert(0, "_unk_")
        self.vocab_list = self.vocab
        self.seq_length = charecter_length
        self.vocab_dict = self.vocab_map(self.vocab)
        self.reverse_vocab_map()

    def tokenize_text(self, text):
        """
        Convert a given text in to a list of integer/embedding tokens
        Args:
            text: a sentence text

        Returns:
            Integer list representation of a text
        """
        text = self.clean_text(text)
        text = list(text)
        if len(text) >= self.seq_length:
            text = text[:self.seq_length]
            text_tokens = [self.vocab_dict[t] for t in text]
        else:
            padding = self.seq_length - len(text)
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tokens.extend([1] * padding)
        return text_tokens


class WordTokenizer(Tokenizer):
    """Load embeddings from a file
    """
    def __init__(self,
                 embeddings_path="/artifacts/pretrained-embeddings/glove-6B" +
                                 "-300d/1.0.0/glove.6B.300d.txt",
                 sequence_length=100):
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
        self.embedding_matrix = np.array([x.split()[1:] for x in vector_file[2:-1]],
                                     dtype="float32")
        self.dim = self.embedding_matrix.shape[1]
        # we will set the _unk_ token to average of all vectors and pad to 0
        self.embedding_average = np.average(self.embedding_matrix, axis=0).reshape(1, -1)
        self.embedding_matrix = np.concatenate((self.embedding_average, np.zeros((1, self.dim)), self.embedding_matrix))

    def fit_vocab(self, text_list):
        X_tokens = []
        embedding_array = []
        response_vocab_list = []
        for txt in text_list:
            X_tokens.append(nltk.word_tokenize(self.clean_text(txt)))
        freq = collections.Counter(p for o in X_tokens for p in o)
        response_words = [o for o, c in freq.most_common(self.max_vocab) if c > self.min_freq]
        for word in response_words:
            if (word in self.vocab_list) and (word not in response_vocab_list):
                response_vocab_list.append(word)
                embedding_array.append(self.embedding_matrix[self.vocab_dict[word]])
        embedding_array = np.array(embedding_array, dtype="float32")
        response_vocab_list.insert(0, "_pad_")
        response_vocab_list.insert(0, "_unk_")
        self.vocab_list = response_vocab_list
        self.vocab_dict = collections.defaultdict(int, {v: k for k, v in enumerate(response_vocab_list)})
        # update embedding average
        self.embedding_average = np.average(embedding_array, axis=0).reshape(1, -1)
        self.embedding_matrix = np.concatenate((self.embedding_average, np.zeros((1, self.dim)), embedding_array))


class WordTokenizerMath(Tokenizer):
    """Load embeddings from a file
    """
    def __init__(self,
                 embeddings_path="/artifacts/pretrained-embeddings/glove-6B" +
                                 "-300d/1.0.0/glove.6B.300d.txt"):
        self.glove_vec_dict = None
        self.seq_length = 100
        self.vocab_dict = {}
        self.dim = None
        self.embedding_average = 0
        self.embedding_matrix = np.nan
        self.embeddings_path = embeddings_path
        self.max_vocab = 40000
        self.min_freq = 2
        self._load_glove_embeddings()
        self.reverse_vocab_map()

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
        self.embedding_matrix = np.array([x.split()[1:] for x in vector_file[2:-1]],
                                     dtype="float32")
        self.dim = self.embedding_matrix.shape[1]
        # we will set the _unk_ token to average of all vectors and pad to 0
        self.embedding_average = np.average(self.embedding_matrix, axis=0).reshape(1, -1)
        self.embedding_matrix = np.concatenate((self.embedding_average, np.zeros((1, self.dim)), self.embedding_matrix))

    def replace_equations(self, text, replace_token="equation"):
        replace_token_len = len(replace_token)
        equation_indices = []
        start_index = -1
        end_index = -1
        text = list(text)
        for idx, t in enumerate(text):
            if t == "$" and start_index == -1:
                start_index = idx
            elif t == "$" and end_index == -1:
                end_index = idx
                equation_indices.append((start_index, end_index))
                start_index = -1
                end_index = -1
        for ei in equation_indices:
            equation_length = ei[1] - ei[0]
            if equation_length > replace_token_len:
                text[ei[0]:ei[1] + 1] = replace_token + " " * ((equation_length - replace_token_len) + 1)
            else:
                text[ei[0]:ei[1] + 1] = replace_token
        text = "".join(text)
        return text

    def numbers_to_text(self, numbers_array):
        out_text = " ".join([self.reverse_vocab_dict[x] for x in numbers_array if x != 1])
        return out_text

    def tokenize_latex_math(self, text):
        text = self.clean_text(text)
        text = self.replace_equations(text)
        text_list = nltk.word_tokenize(text)
        return text_list

    def fit_vocab(self, text_list):
        X_tokens = []
        embedding_array = []
        response_vocab_list = []
        for txt in text_list:
            X_tokens.append(nltk.word_tokenize(self.clean_text(txt)))
        freq = collections.Counter(p for o in X_tokens for p in o)
        response_words = [o for o, c in freq.most_common(self.max_vocab) if c > self.min_freq]
        for word in response_words:
            if (word in self.vocab_list) and (word not in response_vocab_list):
                response_vocab_list.append(word)
                embedding_array.append(self.embedding_matrix[self.vocab_dict[word]])
        embedding_array = np.array(embedding_array, dtype="float32")
        response_vocab_list.insert(0, "_pad_")
        response_vocab_list.insert(0, "_unk_")
        self.vocab_list = response_vocab_list
        self.vocab_dict = collections.defaultdict(int, {v: k for k, v in enumerate(response_vocab_list)})
        # update embedding average
        self.embedding_average = np.average(embedding_array, axis=0).reshape(1, -1)
        self.embedding_matrix = np.concatenate((self.embedding_average, np.zeros((1, self.dim)), embedding_array))

    def tokenize_text(self, text):
        """
        Convert a given text in to a list of integer/embedding tokens
        Args:
            text: a sentence text

        Returns:
            Integer list representation of a text
        """
        text = self.tokenize_latex_math(text)
        if len(text) >= self.seq_length:
            text = text[:self.seq_length]
            text_tokens = [self.vocab_dict[t] for t in text]
        else:
            padding = self.seq_length - len(text)
            text_tokens = [self.vocab_dict[t] for t in text]
            text_tokens.extend([1] * padding)
        return text_tokens


def tokenize_text_scores(token_model,
                         train_text_array,
                         test_text_array,
                         train_scores_array,
                         test_scores_array,
                         test_set_included=True):
    x_train = []
    x_test = []
    for idx, txt in enumerate(train_text_array):
        x_train.append(token_model.tokenize_text(txt))
        if idx % 5000 == 0:
            logging.info("Done with {} examples".format(idx))

    if test_set_included:
        for idx, txt in enumerate(test_text_array):
            x_test.append(token_model.tokenize_text(txt))
            if idx % 50000 == 0:
                logging.info("Done with {} examples".format(idx))
        logging.info(np.unique(test_scores_array))
    logging.info(np.unique(train_scores_array))
    
#     n_unique = len(np.unique(train_scores_array))
#     enc = OneHotEncoder(sparse=False, n_values=n_unique)
#     train_scores_array = enc.fit_transform(train_scores_array.reshape(-1, 1))
#     if test_set_included:
#         test_scores_array = enc.fit_transform(test_scores_array.reshape(-1, 1))
#     if test_set_included:
#         logging.info(
#             str(train_scores_array.shape) + str(test_scores_array.shape))

    return np.array(x_train, dtype="int").reshape(-1, token_model.seq_length), \
           np.array(x_test, dtype="int").reshape(-1, token_model.seq_length), \
           train_scores_array, test_scores_array


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


def iterate_minibatches_two_inputs(inputs1, inputs2, targets, batchsize, shuffle=True):
    """Loads in two arrays and returns a batch generator
    """
    assert len(inputs1) == len(targets) == len(inputs2)
    indices = np.arange(len(inputs1))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt]


def load_asap_data(data_path, set_no, embedding_path, seq_length):
    data = pd.read_csv(data_path)
    data_input = data[data['essay_set'] == set_no]
    
    data_train = data_input[data_input['set'] == 'train']
    data_test = data_input[data_input['set'] == 'dev']
    
    normalizer = MinMaxScaler()
    
    x_train = data_train['essay'].values
    y_train = data_train['score'].values
    y_train = normalizer.fit_transform(y_train.reshape(-1, 1))
    
    x_test = data_test['essay'].values
    y_test = data_test['score'].values
    y_test = normalizer.transform(y_test.reshape(-1, 1))
    
    token_model = WordTokenizer(embeddings_path=embedding_path,sequence_length=seq_length)
    x_train, x_test, y_train, y_test = tokenize_text_scores(token_model,
                                                            x_train,
                                                            x_test,
                                                            y_train,
                                                            y_test)
#     tokenize = text.Tokenizer(num_words=10000, char_level=False)
    
#     tokenize.fit_on_texts(x_train) # only fit on train
#     x_train = tokenize.texts_to_matrix(x_train)
#     x_test = tokenize.texts_to_matrix(x_test)
    
    print(x_train[0])
    random_indices = np.random.permutation(x_train.shape[0])
    logging.info("Permuting data")
    logging.info("Train Shapes")
    logging.info("-" * 50)
    logging.info((x_train.shape,y_train.shape))
    logging.info("Test shape")
    logging.info("-" * 50)
    logging.info((x_test.shape, y_test.shape))
    x_train = x_train[random_indices, :]
    y_train = y_train[random_indices, :]
    
    print('data')
    print(x_train[0:10])
    print(y_train[0:10])
    print(y_test[0:10])
    
    return x_train, x_test, y_train, y_test, token_model, normalizer
    
    
def replace_equations(text, replace_token="equation"):
    replace_token_len = len(replace_token)
    equation_indices = []
    start_index = -1
    end_index = -1
    text = list(text)
    for idx, t in enumerate(text):
        if t == "$" and start_index == -1:
            start_index = idx
        elif t == "$" and end_index == -1:
            end_index = idx
            equation_indices.append((start_index, end_index))
            start_index = -1
            end_index = -1
    print(equation_indices)
    for ei in equation_indices:
        equation_length = ei[1] - ei[0]
        if equation_length > replace_token_len:
            text[ei[0]:ei[1] + 1] = replace_token + " " * ((equation_length - replace_token_len) + 1)
        else:
            text[ei[0]:ei[1] + 1] = replace_token
    text = "".join(text)
    return text
