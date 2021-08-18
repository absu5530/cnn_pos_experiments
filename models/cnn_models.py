#!/usr/bin/env python

import logging

import tensorflow as tf


class CNNBlocks(object):
    def embedding_layer(self, input_x, name, vocab_size, embedding_size,
                        train_embedding=True, embedding_matrix=None,
                        initializer_type="xavier"):
        if embedding_matrix is not None:
            embedding_initializer = tf.constant(embedding_matrix,
                                                dtype="float32")
        else:
            if initializer_type == "random-normal":
                embedding_initializer = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
            elif initializer_type == "xavier":
                embedding_initializer = tf.get_variable("W", shape=[vocab_size, embedding_size],
                                                        initializer=tf.contrib.layers.xavier_initializer())
            elif initializer_type == "he-normal":
                embedding_initializer = tf.get_variable("W", shape=[vocab_size, embedding_size],
                                                        initializer=tf.initializers.he_normal())
        with tf.name_scope(name):
            embedding_dynamic = tf.get_variable(name + "_embedding_W1",
                                                initializer=embedding_initializer,
                                                trainable=train_embedding)

            embedded_chars = tf.nn.embedding_lookup(embedding_dynamic,
                                                    input_x)
            # the output of the below is of shape
            # [batch_size, time, emb, num_channels]
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            return embedded_chars_expanded

    def conv_block(self,
                   input_tensor,
                   phase,
                   filter_shape=[3, 1, 64, 64],
                   name="1",
                   do_batch_norm=False):
        """Define a conv block with two conv layers and 2 batch norm layers
        """
        n_filters = filter_shape[-1]
        filter_shape1 = filter_shape
        filter_stride1 = [1, 1, filter_shape[1], 1]
        with tf.name_scope(name):
            W = tf.get_variable(name=name + "_W1",
                                shape=filter_shape1,
                                initializer=tf.keras.initializers.he_normal())
            # We dont need a bias term here because batch norm takes care of
            # this
            conv1 = tf.nn.conv2d(input_tensor,
                                 W,
                                 strides=filter_stride1,
                                 padding="SAME")
            # add some summaries
            logging.info("Shape of conv1 filters at block: %s " % name)
            logging.info(filter_shape1)
            # this layer takes care of the entire operation
            if do_batch_norm:
                logging.info("Performing batch normalization1")
                conv1 = tf.contrib.layers.batch_norm(conv1,
                                                     decay=0.9999,
                                                     center=True,
                                                     scale=True,
                                                     is_training=phase,
                                                     zero_debias_moving_mean=True)

            activations_1 = tf.nn.relu(conv1)
            logging.info("Shape of outputs from conv1 block: %s " % name)
            logging.info(activations_1.get_shape())
            # the -1 here represents the input channels
            # NHWC
            filter_shape2 = [3, 1, filter_shape1[-1], n_filters]
            filter_stride2 = [1, 1, filter_shape2[1], 1]

            W = tf.get_variable(name=name + "_W2",
                                shape=filter_shape2,
                                initializer=tf.keras.initializers.he_normal())
            logging.info("Shape of conv2 filters at block: %s " % name)
            logging.info(filter_shape2)
            conv2 = tf.nn.conv2d(activations_1,
                                 W,
                                 strides=filter_stride2,
                                 padding="SAME")
            if do_batch_norm:
                logging.info("Performing batch normalization2")
                conv2 = tf.contrib.layers.batch_norm(conv2,
                                                     decay=0.9999,
                                                     center=True,
                                                     scale=True,
                                                     is_training=phase,
                                                     zero_debias_moving_mean=True)
            activations_2 = tf.nn.relu(conv2)
            logging.info("Shape of outputs from conv2 block: %s " % name)
            logging.info(activations_2.get_shape())
            return activations_2

    def fc_layer(self, in_layer, fc_units, name):
        """A fully connected layer block, when in layer comes in it must be
        reshaped
        """
        with tf.name_scope(name):
            # get the shape of the incoming layer
            shape = in_layer.get_shape().as_list()
            dim = 1
            # We don't want to consider the first int in the list because this
            # is the batch size
            for d in shape[1:]:
                dim *= d

            in_layer = tf.reshape(in_layer, [-1, dim])
            # Initialize the weights and bias
            weights, bias = self.get_fc_weights_bias(name, dim, fc_units)
            # Do the affine transformation
            fc = tf.matmul(in_layer, weights) + bias
            # point-wise non-linearity
            fc = tf.nn.relu(fc)
            logging.info("Shape of weights and bias of fc layer %s " % name)
            logging.info((dim, fc_units))
            return fc

    def get_fc_weights_bias(self, name, in_shape, fc_shape):
        """A function to initialize the weights and bias of a fully connected
        layer of a neural network
        """
        W = tf.get_variable(name=name + "_W",
                            shape=[in_shape, fc_shape],
                            initializer=tf.keras.initializers.he_normal())

        b = tf.get_variable(name=name + "_b",
                            shape=[fc_shape],
                            initializer=tf.keras.initializers.he_normal())
        return W, b

    def cnn_and_pooling_layer(self, input_tensor, layer_no, FILTER_SIZE,
                              EMBEDDING_SIZE_WORDS, N_FILTERS, KSIZE, STRIDES,
                              do_top_k, N_CHANNELS):
        logging.info("Shape input tensor")
        logging.info(self.input_tensor.get_shape())

        with tf.name_scope("conv_words" + str(layer_no)):
            # The filter shape should be of
            # [filter_size, EMBEDDING_SIZE, n_channels, N_FILTERS] where
            # n_channels should be incoming number of channels
            filter_shape = [FILTER_SIZE, EMBEDDING_SIZE_WORDS, N_CHANNELS,
                            N_FILTERS]
            W = tf.get_variable(name='conv_words_W' + str(layer_no),
                                shape=filter_shape,
                                initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable(name='conv_words_b' + str(layer_no),
                                shape=[N_FILTERS],
                                initializer=tf.keras.initializers.he_normal())
            self.conv_words = tf.nn.conv2d(input_tensor,
                                           W,
                                           strides=[1, 1, EMBEDDING_SIZE_WORDS,
                                                    1],
                                           padding="SAME",
                                           name="conv_words" + str(layer_no))

        self.rel_conv_words = tf.nn.relu(tf.nn.bias_add(self.conv_words, b),
                                         name="conv_words_relu")

        logging.info("Shape after rel")
        logging.info(self.rel_conv_words.get_shape())

        if do_top_k and layer_no == 2:
            logging.info("Performing top K")
            self.conv_reshape_words = tf.transpose(self.rel_conv_words,
                                                   [0, 3, 2, 1])
            logging.info(("After reshaping: ", self.conv_reshape_words))
            with tf.name_scope("max_k_words"):
                self.pool_words, indices = tf.nn.top_k(self.conv_reshape_words,
                                                       k=3,
                                                       sorted=False)
            logging.info("Shape of top K")
            logging.info(self.pool_words.shape)
        else:
            with tf.name_scope("max_pool"):
                # For input with [batch_size, time, emb, num_channels] we do
                # temporal max pool on the time dimension with filter size of 3
                # and stride of 2 this should reduce the temporal dimension by 2
                self.pool_words = tf.nn.max_pool(self.rel_conv_words,
                                                 ksize=[1, KSIZE, 1, 1],
                                                 strides=[1, STRIDES, 1, 1],
                                                 padding="VALID")

        logging.info("Shape after pooling")
        logging.info(self.pool_words.get_shape())
        return self.pool_words


class DeepGenericMaxPooling(CNNBlocks):
    pass


class ShallowWordCNN(CNNBlocks):
    def __init__(self,
                 ESSAY_LENGTH_WORDS=500,
                 VOCAB_SIZE_WORDS=40000,
                 EMBEDDING_SIZE_WORDS=50,
                 VOCAB_SIZE_POS=45,
                 INITIALIZER_TYPE="xavier",
                 EMBEDDING_SIZE_POS=50,
                 do_batch_norm=False,
                 do_dropout=False,
                 dropout_rate=0.3,
                 num_fc_layers=2,
                 do_top_k=False,
                 train_word_embedding=True,
                 train_pos_embedding=True,
                 embedding_matrix=None,
                 N_FILTERS=10,
                 FILTER_SIZE_1=2,
                 FILTER_SIZE_2=3,
                 FILTER_SIZE_3=4,
                 STRIDES=3,
                 KSIZE=3,
                 N_CHANNELS=2):
        self.input_x = tf.placeholder(tf.int32,
                                      [None, ESSAY_LENGTH_WORDS, 2],
                                      name="input_x")

        self.input_y = tf.placeholder(tf.float32,
                                      [None, 1], name="input_y")
        self.phase = tf.placeholder(tf.bool, name='phase')
        # we want to deal with multiple channels of information. This can be
        # a channel of static embeddings, a channel of dynamic embeddings
        self.embedded_words_expanded = self.embedding_layer(
            input_x=self.input_x[:, :, 0],
            name="word_embedding",
            vocab_size=VOCAB_SIZE_WORDS,
            embedding_size=EMBEDDING_SIZE_WORDS,
            train_embedding=train_word_embedding,
            initializer_type=INITIALIZER_TYPE,
            embedding_matrix=embedding_matrix)

        self.input_tensor = self.embedded_words_expanded
        print("Shape of input tensor")
        print(self.input_tensor.get_shape())

        self.pool0_words = self.cnn_and_pooling_layer(self.input_tensor, 0, FILTER_SIZE_1, EMBEDDING_SIZE_WORDS,
                                                      N_FILTERS, KSIZE, STRIDES, do_top_k, 1)

        self.pool1_words = self.cnn_and_pooling_layer(self.input_tensor, 1, FILTER_SIZE_2, EMBEDDING_SIZE_WORDS, N_FILTERS, KSIZE,
                                                      STRIDES, do_top_k, 1)

        self.pool2_words = self.cnn_and_pooling_layer(self.input_tensor, 2, FILTER_SIZE_3, EMBEDDING_SIZE_WORDS,
                                                      N_FILTERS, KSIZE, STRIDES, do_top_k, 1)

        # concatenation by correct axis
        self.conv_concat = tf.concat((self.pool0_words, self.pool1_words, self.pool2_words),
                                     axis=1)
        logging.info("Shape of concat layer")
        logging.info(self.conv_concat.get_shape())

        # Build regression layer
        with tf.name_scope("output"):
            self.concat_dropout = tf.layers.dropout(inputs=self.conv_concat,
                                                    rate=dropout_rate,
                                                    seed=89312,
                                                    training=self.phase,
                                                    )
            self.flattened_concat = tf.layers.flatten(self.concat_dropout)
            # self.penultimate = tf.layers.dense(self.flattened_concat, 4)
            self.predictions = tf.layers.dense(self.flattened_concat, 1, name="predictions",
                                               kernel_initializer=tf.keras.initializers.he_normal(),
                                               activation=tf.nn.sigmoid)

        print('Shape of predictions')
        print(self.predictions.name)
        print(self.predictions.get_shape())

        self.loss = tf.losses.mean_squared_error(tf.to_float(self.input_y), self.predictions)


class DeepWordOneHotPosCNN(CNNBlocks):
    def __init__(self,
                 ESSAY_LENGTH_WORDS=500,
                 VOCAB_SIZE_WORDS=40000,
                 EMBEDDING_SIZE_WORDS=50,
                 VOCAB_SIZE_POS=45,
                 INITIALIZER_TYPE="xavier",
                 EMBEDDING_SIZE_POS=50,
                 do_batch_norm=False,
                 do_dropout=False,
                 dropout_rate=0.3,
                 num_fc_layers=2,
                 do_top_k=False,
                 train_word_embedding=True,
                 train_pos_embedding=True,
                 embedding_matrix=None,
                 N_CHANNELS=2):
        self.input_x = tf.placeholder(tf.int32,
                                      [None, ESSAY_LENGTH_WORDS, EMBEDDING_SIZE_WORDS, 2],
                                      name="input_x")

        self.input_y = tf.placeholder(tf.float32,
                                      [None, 1], name="input_y")
        self.phase = tf.placeholder(tf.bool, name='phase')
        # we want to deal with multiple channels of information. This can be
        # a channel of static embeddings, a channel of dynamic embeddings
        self.embedded_words_expanded = self.embedding_layer(
            input_x=self.input_x[:, :, 0, 0],
            name="word_embedding",
            vocab_size=VOCAB_SIZE_WORDS,
            embedding_size=EMBEDDING_SIZE_WORDS,
            train_embedding=train_word_embedding,
            initializer_type=INITIALIZER_TYPE,
            embedding_matrix=embedding_matrix)
        # Last channel will be POS
        if N_CHANNELS > 1:
            self.embedded_pos_expanded = self.embedding_layer(
                input_x=self.input_x[:, :, 0, -1],
                name="pos_embeddings",
                vocab_size=VOCAB_SIZE_POS,
                embedding_size=EMBEDDING_SIZE_POS,
                train_embedding=train_pos_embedding,
                embedding_matrix=None,
                initializer_type=INITIALIZER_TYPE
            )
        self.input_tensor = tf.concat([self.embedded_words_expanded,
                                       self.embedded_pos_expanded],
                                      axis=3)
        print("Shape of input tensor")
        print(self.input_tensor.get_shape())

        self.pool1_words = self.cnn_and_pooling_layer(self.input_tensor, 1, 3, EMBEDDING_SIZE_WORDS, 8, 3, 2, do_top_k,
                                                      N_CHANNELS)

        self.pool2_words = self.cnn_and_pooling_layer(self.pool1_words, 2, 3, EMBEDDING_SIZE_WORDS, 8, 3, 2, do_top_k,
                                                      8)

        self.pool3_words = self.cnn_and_pooling_layer(self.pool2_words, 3, 3, EMBEDDING_SIZE_WORDS, 16, 3, 2, do_top_k,
                                                      8)

        # Build regression layer
        with tf.name_scope("output"):
            self.concat_dropout = tf.layers.dropout(inputs=self.pool3_words,
                                                    rate=dropout_rate,
                                                    seed=89312,
                                                    training=self.phase,
                                                    )
            self.flattened_concat = tf.layers.flatten(self.concat_dropout)
            # self.penultimate = tf.layers.dense(self.flattened_concat, 4)
            self.predictions = tf.layers.dense(self.flattened_concat, 1, name="predictions",
                                               kernel_initializer=tf.keras.initializers.he_normal(),
                                               activation=tf.nn.sigmoid)

        print('Shape of predictions')
        print(self.predictions.name)
        print(self.predictions.get_shape())

        self.loss = tf.losses.mean_squared_error(tf.to_float(self.input_y), self.predictions)


class DeepWordPosCNN(CNNBlocks):
    def __init__(self,
                 ESSAY_LENGTH_WORDS=500,
                 VOCAB_SIZE_WORDS=40000,
                 EMBEDDING_SIZE_WORDS=50,
                 VOCAB_SIZE_POS=45,
                 INITIALIZER_TYPE="xavier",
                 EMBEDDING_SIZE_POS=50,
                 do_batch_norm=False,
                 do_dropout=False,
                 dropout_rate=0.3,
                 num_fc_layers=2,
                 do_top_k=False,
                 train_word_embedding=True,
                 train_pos_embedding=True,
                 embedding_matrix=None,
                 pos_embedding_matrix=None,
                 N_CHANNELS=2):
        self.input_x = tf.placeholder(tf.int32,
                                      [None, ESSAY_LENGTH_WORDS, 2],
                                      name="input_x")

        self.input_y = tf.placeholder(tf.float32,
                                      [None, 1], name="input_y")
        self.phase = tf.placeholder(tf.bool, name='phase')
        # we want to deal with multiple channels of information. This can be
        # a channel of static embeddings, a channel of dynamic embeddings

        self.embedded_words_expanded = self.embedding_layer(
            input_x=self.input_x[:, :, 0],
            name="word_embedding",
            vocab_size=VOCAB_SIZE_WORDS,
            embedding_size=EMBEDDING_SIZE_WORDS,
            train_embedding=train_word_embedding,
            initializer_type=INITIALIZER_TYPE,
            embedding_matrix=embedding_matrix)
        # Last channel will be POS
        if N_CHANNELS > 1:
            self.embedded_pos_expanded = self.embedding_layer(
                input_x=self.input_x[:, :, -1],
                name="pos_embeddings",
                vocab_size=VOCAB_SIZE_POS,
                embedding_size=EMBEDDING_SIZE_POS,
                train_embedding=train_pos_embedding,
                embedding_matrix=pos_embedding_matrix,
                initializer_type=INITIALIZER_TYPE
            )

        if N_CHANNELS > 1:
            self.input_tensor = tf.concat([self.embedded_words_expanded,
                                           self.embedded_pos_expanded],
                                          axis=3)
        else:
            self.input_tensor = self.embedded_words_expanded
        print("Shape of input tensor")
        print(self.input_tensor.get_shape())

        self.pool1_words = self.cnn_and_pooling_layer(self.input_tensor, 1, 3, EMBEDDING_SIZE_WORDS, 8, 3, 2, do_top_k,
                                                      N_CHANNELS)

        self.pool2_words = self.cnn_and_pooling_layer(self.pool1_words, 2, 3, EMBEDDING_SIZE_WORDS, 8, 3, 2, do_top_k,
                                                      8)

        self.pool3_words = self.cnn_and_pooling_layer(self.pool2_words, 3, 3, EMBEDDING_SIZE_WORDS, 16, 3, 2, do_top_k,
                                                      8)

        # Build regression layer
        with tf.name_scope("output"):
            self.concat_dropout = tf.layers.dropout(inputs=self.pool3_words,
                                                    rate=dropout_rate,
                                                    seed=89312,
                                                    training=self.phase,
                                                    )
            self.flattened_concat = tf.layers.flatten(self.concat_dropout)
            # self.penultimate = tf.layers.dense(self.flattened_concat, 4)
            self.predictions = tf.layers.dense(self.flattened_concat, 1, name="predictions",
                                               kernel_initializer=tf.keras.initializers.he_normal(),
                                               activation=tf.nn.sigmoid)

        print('Shape of predictions')
        print(self.predictions.name)
        print(self.predictions.get_shape())

        self.loss = tf.losses.mean_squared_error(tf.to_float(self.input_y), self.predictions)
