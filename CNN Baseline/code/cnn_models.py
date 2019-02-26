#!/usr/bin/env python

import tensorflow as tf
import logging


class CNNBlocks(object):
    def embedding_layer(self, input_x, name, vocab_size, embedding_size, train_embedding=True, embedding_matrix=None):
        if not train_embedding:
            embedding_initializer = tf.constant(embedding_matrix,
                                                dtype="float32")
        else:
            embedding_initializer = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
        with tf.name_scope(name):
            embedding_dynamic = tf.get_variable(name + "_embedding_W1",
                                                 initializer= embedding_initializer,
                                                 trainable=train_embedding)

            embedded_chars = tf.nn.embedding_lookup(embedding_dynamic,
                                                    input_x)
            # the output of the below is of shape [batch_size, time, emb, num_channels]
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            return embedded_chars_expanded


    def conv_block(self, input_tensor, phase, filter_shape=[3, 1, 64, 64],
                   name="1", do_batchnorm = False):
        """Define a conv block with two conv layers and 2 batch norm layers
        """
        n_filters = filter_shape[-1]
        filter_shape1 = filter_shape
        filter_stride1 = [1, 1, filter_shape[1], 1]
        with tf.name_scope(name):
            W = tf.get_variable(name= name + "_W1",
                                shape= filter_shape1,
                                initializer= tf.keras.initializers.he_normal())
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
            if do_batchnorm:
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

            W = tf.get_variable(name= name + "_W2",
                                shape= filter_shape2,
                                initializer= tf.keras.initializers.he_normal())
            logging.info("Shape of conv2 filters at block: %s " % name)
            logging.info(filter_shape2)
            conv2 = tf.nn.conv2d(activations_1,
                                 W,
                                 strides=filter_stride2,
                                 padding="SAME")
            if do_batchnorm:
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
                            shape = [in_shape, fc_shape],
                            initializer=tf.keras.initializers.he_normal())

        b = tf.get_variable(name=name + "_b",
                            shape=[fc_shape],
                            initializer=tf.keras.initializers.he_normal())
        return W, b


class CNN_baseline(CNNBlocks):
    def __init__(self,
                 ESSAY_LENGTH_WORDS=500,
                 VOCAB_SIZE_WORDS=40000,
                 EMBEDDING_SIZE_WORDS=50,
                 do_batch_norm=False,
                 do_dropout=False,
                 dropout_rate=0.3,
                 num_fc_layers = 2,
                 do_top_k=False,
                 train_embedding=True,
                 embedding_matrix=None,normalizer=None):
        self.input_x = tf.placeholder(tf.int32,
                                            [None, ESSAY_LENGTH_WORDS],
                                            name="input_x_words")
        
        logging.info("Shape of input x")
        logging.info(self.input_x.get_shape())
        
        self.input_y = tf.placeholder(tf.float32,
                                      [None,1], name="input_y")
        self.do_batch_norm = do_batch_norm
        # For batch norm we need to know if this is train or the test phase
        self.phase = tf.placeholder(tf.bool, name='phase')

        self.embedded_words_expanded = self.embedding_layer(
                                                            input_x=self.input_x,
                                                            name="word_embedding",
                                                            vocab_size=VOCAB_SIZE_WORDS,
                                                            embedding_size=EMBEDDING_SIZE_WORDS,
                                                            train_embedding=train_embedding,
                                                            embedding_matrix=embedding_matrix)

        logging.info("Shape of input word representation")
        logging.info(self.embedded_words_expanded.get_shape())


        # the first convolution layer of the network
        
        # Word level model
        # the first convolution layer of the network
        with tf.name_scope("conv0_words"):
            # The filter shape should be of
            # [filter_size, EMBEDDING_SIZE, n_channels, N_FILTERS] where
            # n_channels should be incoming number of channels
            filter_shape = [2, EMBEDDING_SIZE_WORDS, 1, 2]
            W = tf.get_variable(name='conv0_words_W',
                                shape=filter_shape,
                                initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable(name="conv0_words_b",
                                shape=[64],
                                initializer=tf.keras.initializers.he_normal())
            self.conv0_words = tf.nn.conv2d(self.embedded_words_expanded,
                                W,
                                strides=[1, 1, EMBEDDING_SIZE_WORDS, 1],
                                padding="SAME",
                                            name="conv0_words")
        
        self.rel_conv0_words = tf.nn.relu(self.conv0_words)
        
        with tf.name_scope("max_pool0"):
            # For input with [batch_size, time, emb, num_channels] we do
            # temporal max pool on the time dimension with filter size of 3
            # and stride of 2 this should reduce the temporal dimension by 2
            self.pool0_words = tf.nn.max_pool(self.rel_conv0_words,
                                             ksize=[1, 2, 2, 1],
                                             strides=[1, 1, 1, 1],
                                             padding="SAME")

        logging.info("Shape after pooling 0")
        logging.info(self.pool0_words.get_shape())
        
        
        with tf.name_scope("conv1_words"):
            # The filter shape should be of
            # [filter_size, EMBEDDING_SIZE, n_channels, N_FILTERS] where
            # n_channels should be incoming number of channels
            filter_shape = [3, EMBEDDING_SIZE_WORDS, 1, 2]
            W = tf.get_variable(name='conv1_words_W',
                                shape=filter_shape,
                                initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable(name="conv1_words_b",
                                shape=[64],
                                initializer=tf.keras.initializers.he_normal())
            self.conv1_words = tf.nn.conv2d(self.embedded_words_expanded,
                                W,
                                strides=[1, 1, EMBEDDING_SIZE_WORDS, 1],
                                padding="SAME",
                                            name="conv1_words")
        
        logging.info("Shape after conv1 1")
        logging.info(self.conv1_words.get_shape()) 
        
        self.rel_conv1_words = tf.nn.relu(self.conv1_words)
            
        with tf.name_scope("max_pool1"):
            # For input with [batch_size, time, emb, num_channels] we do
            # temporal max pool on the time dimension with filter size of 3
            # and stride of 2 this should reduce the temporal dimension by 2
            self.pool1_words = tf.nn.max_pool(self.rel_conv1_words,
                                             ksize=[1, 2, 2, 1],
                                             strides=[1, 1, 1, 1],
                                             padding="SAME")

        logging.info("Shape after pooling 1")
        logging.info(self.pool1_words.get_shape())    
        
        
        with tf.name_scope("conv2_words"):
            # The filter shape should be of
            # [filter_size, EMBEDDING_SIZE, n_channels, N_FILTERS] where
            # n_channels should be incoming number of channels
            filter_shape = [4, EMBEDDING_SIZE_WORDS, 1, 2]
            W = tf.get_variable(name='conv2_words_W',
                                shape=filter_shape,
                                initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable(name="conv2_words_b",
                                shape=[64],
                                initializer=tf.keras.initializers.he_normal())
            self.conv2_words = tf.nn.conv2d(self.embedded_words_expanded,
                                W,
                                strides=[1, 1, EMBEDDING_SIZE_WORDS, 1],
                                padding="SAME",
                                            name="conv2_words")
            
        self.rel_conv2_words = tf.nn.relu(self.conv2_words)

        
        with tf.name_scope("max_pool2"):
            # For input with [batch_size, time, emb, num_channels] we do
            # temporal max pool on the time dimension with filter size of 3
            # and stride of 2 this should reduce the temporal dimension by 2
            self.pool2_words = tf.nn.max_pool(self.rel_conv2_words,
                                             ksize=[1, 2, 2, 1],
                                             strides=[1, 1, 1, 1],
                                             padding="SAME")

        logging.info("Shape after pooling 2")
        logging.info(self.pool2_words.get_shape())  
        
        #concatenation by correct axis
        self.conv_concat = tf.concat((self.pool0_words, self.pool1_words, self.pool2_words),
                                          axis=1)
        logging.info("Shape of concat layer")
        logging.info(self.conv_concat.get_shape())
        
        
        
        #Build regression layer
        
        with tf.name_scope("output"):
            self.flattened_concat = tf.layers.flatten(self.conv_concat)
            self.predictions= tf.layers.dense(self.flattened_concat, 1, name="predictions", kernel_initializer=tf.keras.initializers.he_normal(),activation=tf.nn.sigmoid)
            
        
#         with tf.name_scope("output"):
#             self.fc_layer = tf.layers.dense(tf.to_float(self.input_x), 256, activation=tf.nn.relu)
#             #self.flattened_concat = tf.layers.flatten(self.fc_layer)
#             self.predictions= tf.layers.dense(self.fc_layer, 1, name="predictions", kernel_initializer=tf.keras.initializers.he_normal(),activation=tf.nn.sigmoid)
            
        
        print('Shape of predictions')
        print(self.predictions.name)
        print(self.predictions.get_shape())
        
        self.loss = tf.losses.mean_squared_error(tf.to_float(self.input_y),self.predictions)
        #self.loss = tf.reduce_mean(tf.squared_difference(self.predictions, tf.to_float(self.input_y)))
        
#         with tf.name_scope("accuracy"):
#             correct_predictions = tf.equal(self.predictions,self.input_y)

#             self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
#                                                    "float32"),
#                                            name="accuracy")
#             tf.summary.scalar("accuracy", self.accuracy)
   

class TextCNN_1_char_words(CNNBlocks):
    def __init__(self,
                 ESSAY_LENGTH_CHAR=1024,
                 ESSAY_LENGTH_WORDS=100,
                 N_SCORES=4,
                 VOCAB_SIZE_CHAR=71,
                 EMBEDDING_SIZE_CHAR=16,
                 VOCAB_SIZE_WORDS=40000,
                 EMBEDDING_SIZE_WORDS=300,
                 do_batch_norm=False,
                 do_dropout=False,
                 dropout_rate=0.3,
                 num_fc_layers = 2,
                 do_top_k=False,
                 train_embedding=True,
                 embedding_matrix=None):
        self.input_x_char = tf.placeholder(tf.int32,
                                           [None, ESSAY_LENGTH_CHAR],
                                           name="input_x_char")
        self.input_x_words = tf.placeholder(tf.int32,
                                            [None, ESSAY_LENGTH_WORDS],
                                            name="input_x_words")

        self.input_y = tf.placeholder(tf.float32,
                                      [None, N_SCORES], name="input_y")
        self.do_batch_norm = do_batch_norm
        # For batch norm we need to know if this is train or the test phase
        self.phase = tf.placeholder(tf.bool, name='phase')
        # Build a character embedding layer
        self.do_dropout = do_dropout
        self.do_top_k = do_top_k

        self.embedded_chars_expanded = self.embedding_layer(
                                                            input_x=self.input_x_char,
                                                            name="char_embedding",
                                                            vocab_size=VOCAB_SIZE_CHAR,
                                                            embedding_size=EMBEDDING_SIZE_CHAR,
                                                            train_embedding=True,
                                                            embedding_matrix=None)

        self.embedded_words_expanded = self.embedding_layer(
                                                            input_x=self.input_x_words,
                                                            name="word_embedding",
                                                            vocab_size=VOCAB_SIZE_CHAR,
                                                            embedding_size=VOCAB_SIZE_WORDS,
                                                            train_embedding=train_embedding,
                                                            embedding_matrix=embedding_matrix)

        logging.info("Shape of input character representation")
        logging.info(self.embedded_chars_expanded.get_shape())
        logging.info("Shape of input word representation")
        logging.info(self.embedded_words_expanded.get_shape())
        #########################################################################################
        # Character level model
        # the first convolution layer of the network
        with tf.name_scope("conv0_char"):
            # The filter shape should be of
            # [filter_size, EMBEDDING_SIZE, n_channels, N_FILTERS] where
            # n_channels should be incoming number of channels
            filter_shape = [3, EMBEDDING_SIZE_CHAR, 1, 64]
            W = tf.get_variable(name='W',
                                shape=filter_shape,
                                initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable(name="b",
                                shape=[64],
                                initializer=tf.keras.initializers.he_normal())
            self.conv0_char = tf.nn.conv2d(self.embedded_chars_expanded,
                                W,
                                strides=[1, 1, EMBEDDING_SIZE_CHAR, 1],
                                padding="SAME")

        logging.info("Shape of conv1 filters")
        logging.info(filter_shape)
        logging.info("Shape of outputs after conv1")
        logging.info(self.conv0_char.get_shape())
        self.conv0_char = tf.nn.relu(tf.nn.bias_add(self.conv0_char, b), name="conv0_char_relu")

        self.conv1_char = self.conv_block(self.conv0_char,
                                          self.phase,
                                          filter_shape=[3, 1, 64, 64],
                                          name="conv_block_1_char",
                                          do_batchnorm=do_batch_norm)

        with tf.name_scope("max_pool1_char"):
            # For input with [batch_size, time, emb, num_channels] we do
            # temporal max pool on the time dimension with filter size of 3
            # and stride of 2 this should reduce the temporal dimension by 2
            self.pool1_char = tf.nn.max_pool(self.conv1_char,
                                             ksize=[1, 3, 1, 1],
                                             strides=[1, 2, 1, 1],
                                             padding="SAME")

        logging.info("Shape after pooling 1")
        logging.info(self.pool1_char.get_shape())

        self.conv2_char = self.conv_block(self.pool1_char,
                                     self.phase,
                                     filter_shape=[3, 1, 64, 128],
                                     name="conv_block_2_char",
                                     do_batchnorm=do_batch_norm)
        if self.do_top_k:
            logging.info("Performing top K")
            self.conv2_reshape_char = tf.transpose(self.conv2_char, [0, 3, 2, 1])
            logging.info(("After reshaping: ", self.conv2_reshape_char))
            with tf.name_scope("max_k_char"):
                self.pool2_char, indices = tf.nn.top_k(self.conv2_reshape_char,
                                                  k=8,
                                                  sorted=False)
            logging.info("Shape of top K")
            logging.info(self.pool2_char.shape)
        else:
            logging.info("Performing max pool")
            with tf.name_scope("max_pool2_char"):
                # For input with [batch_size, time, emb, num_channels] we do
                # temporal max pool on the time dimension with filter size of 3
                # and stride of 2 this should reduce the temporal dimension by 2
                self.pool2_char = tf.nn.max_pool(self.conv2_char,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 2, 1, 1],
                                            padding="SAME")
            logging.info("Shape after pool 2")
            logging.info(self.pool2_char.get_shape())
        if self.do_dropout:
            logging.info("Performing dropout after max pool")
            self.pool2_char = tf.layers.dropout(inputs=self.pool2_char,
                                                rate=dropout_rate,
                                                seed=89312,
                                                training=self.phase,
                                                )
        self.fc_char = self.fc_layer(self.pool2_char, 2048,
                                     "fc_char_{}".format(0))
        #########################################################################################

        #########################################################################################
        # Word level model
        # the first convolution layer of the network
        with tf.name_scope("conv0_words"):
            # The filter shape should be of
            # [filter_size, EMBEDDING_SIZE, n_channels, N_FILTERS] where
            # n_channels should be incoming number of channels
            filter_shape = [3, EMBEDDING_SIZE_WORDS, 1, 64]
            W = tf.get_variable(name='conv0_words_W',
                                shape=filter_shape,
                                initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable(name="conv0_words_b",
                                shape=[64],
                                initializer=tf.keras.initializers.he_normal())
            self.conv0_words = tf.nn.conv2d(self.embedded_words_expanded,
                                W,
                                strides=[1, 1, EMBEDDING_SIZE_WORDS, 1],
                                padding="SAME",
                                            name="conv0_words")

        logging.info("Shape of conv1 filters")
        logging.info(filter_shape)
        logging.info("Shape of outputs after conv1")
        logging.info(self.conv0_words.get_shape())
        self.conv0_words = tf.nn.relu(tf.nn.bias_add(self.conv0_words, b),
                                     name="conv0_words_relu")

        self.conv1_words = self.conv_block(self.conv0_words,
                                          self.phase,
                                          filter_shape=[3, 1, 64, 64],
                                          name="conv_block_1_words",
                                          do_batchnorm=do_batch_norm)

        with tf.name_scope("max_pool1_char"):
            # For input with [batch_size, time, emb, num_channels] we do
            # temporal max pool on the time dimension with filter size of 3
            # and stride of 2 this should reduce the temporal dimension by 2
            self.pool1_words = tf.nn.max_pool(self.conv1_words,
                                             ksize=[1, 3, 1, 1],
                                             strides=[1, 2, 1, 1],
                                             padding="SAME")

        logging.info("Shape after pooling 1")
        logging.info(self.pool1_words.get_shape())

        self.conv2_words = self.conv_block(self.pool1_words,
                                     self.phase,
                                     filter_shape=[3, 1, 64, 128],
                                     name="conv_block_2_words",
                                     do_batchnorm=do_batch_norm)
        if self.do_top_k:
            logging.info("Performing top K")
            self.conv2_reshape_words = tf.transpose(self.conv2_words, [0, 3, 2, 1])
            logging.info(("After reshaping: ", self.conv2_reshape_words))
            with tf.name_scope("max_k_words"):
                self.pool2_words, indices = tf.nn.top_k(self.conv2_reshape_words,
                                                  k=8,
                                                  sorted=False)
            logging.info("Shape of top K")
            logging.info(self.pool2_words.shape)
        else:
            logging.info("Performing max pool")
            with tf.name_scope("max_pool2_words"):
                # For input with [batch_size, time, emb, num_channels] we do
                # temporal max pool on the time dimension with filter size of 3
                # and stride of 2 this should reduce the temporal dimension by 2
                self.pool2_words = tf.nn.max_pool(self.conv2_words,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 2, 1, 1],
                                            padding="SAME")
            logging.info("Shape after pool 2")
            logging.info(self.pool2_words.get_shape())
        if self.do_dropout:
            logging.info("Performing dropout after max pool")
            self.pool2_words = tf.layers.dropout(inputs=self.pool2_words,
                                                rate=dropout_rate,
                                                seed=89312,
                                                training=self.phase,
                                                )
        self.fc_words = self.fc_layer(self.pool2_words,
                                         2048,
                                         "fc_words_{}".format(0))
        #########################################################################################
        logging.info("shape of char and word models")
        logging.info(self.fc_char.get_shape())
        logging.info(self.fc_words.get_shape())
        # Concat layer for characters and words
        self.word_char_concat = tf.concat((self.fc_char, self.fc_words),
                                          axis=1)
        logging.info("Shape of concat layer")
        logging.info(self.word_char_concat.get_shape())
        if self.do_dropout:
            self.word_char_concat = tf.layers.dropout(inputs=self.word_char_concat,
                                                      rate=dropout_rate * 1.5,
                                                      seed=89312,
                                                      training=self.phase,
                                                      )

        self.fc = self.fc_layer(self.word_char_concat,
                                2048,
                                "fc_word_char")
        # Build the soft max layer one used for prediction an another for
        # optimization
        # make this a function
        with tf.name_scope("output"):
            W = tf.get_variable("softmax_W",
                                shape=[2048, N_SCORES],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[N_SCORES]), name="softmax_b")
            self.scores = tf.nn.xw_plus_b(self.fc, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        print('Shape of scores')
        print(self.scores.get_shape())
        self.soft_prob = tf.nn.softmax(self.scores, -1, "soft_max")

        # Feed the softmax scores and the true labels to the cross entropy
        # functions
        with tf.name_scope("loss"):
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                              labels=self.input_y)
            self.loss = tf.reduce_mean(entropy)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   "float32"),
                                           name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)


# python train_cnn.py --data-set parcc-math-text --model-type both --train-prompt --prompt M500200 --experiment-name parcc_math_cnn1_words_char_model_max_pool_trait_B_test --data-path data/parcc_mpt_M500200_5k_sample.csv --do-dropout --cnn-model cnn-1 --trait B --dropout-rate 0.6