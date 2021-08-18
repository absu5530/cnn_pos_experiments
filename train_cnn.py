#!/usr/bin/env python

import _pickle as pkl
import argparse
import datetime
import json
import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import time
from dataloader import data_loader
from models import cnn_models


def find_meta_file(file_path):
    """There will be only one .meta file in the checkpoints directory.
    It depends on the epoch of early stopping.
    :param file_path: path to model data
    :return: .meta file
    """
    all_files = os.listdir(os.path.join(file_path, "checkpoints"))
    meta_file = list(filter(lambda x: x[-5:] == ".meta", all_files))[0]
    return meta_file


class TrainModel(object):
    def __init__(self):
        self.args = args
        self.x_train, self.x_test, self.y_train, self.y_test, self.token_model = None, None, None, None, None

    def get_cnn_model_object(self):
        if self.args.model_type == "shallow":
            logging.info("Compiling {} model".format(self.args.model_type))
            cnn_mdl = cnn_models.ShallowWordCNN(
                ESSAY_LENGTH_WORDS=args.seq_length,
                VOCAB_SIZE_WORDS=len(self.token_model.vocab_list),
                VOCAB_SIZE_POS=len(self.token_model.pos_tags_list),
                INITIALIZER_TYPE=args.initializer_type,
                EMBEDDING_SIZE_WORDS=args.embedding_size,
                EMBEDDING_SIZE_POS=args.embedding_size,
                do_batch_norm=args.do_batchnorm,
                do_dropout=args.do_dropout,
                dropout_rate=args.dropout_rate,
                num_fc_layers=args.n_fc_layers,
                do_top_k=args.do_topk,
                train_word_embedding=False,
                train_pos_embedding=True,
                embedding_matrix=self.token_model.embedding_matrix,
                N_FILTERS=args.n_filters,
                FILTER_SIZE_1=args.filter_size_1,
                FILTER_SIZE_2=args.filter_size_2,
                FILTER_SIZE_3=args.filter_size_3,
                STRIDES=args.strides,
                KSIZE=args.ksize,
                N_CHANNELS=args.n_channels
            )
        if self.args.model_type == "deep-onehot":
            logging.info("Compiling {} model".format(self.args.model_type))
            cnn_mdl = cnn_models.DeepWordOneHotPosCNN(
                ESSAY_LENGTH_WORDS=args.seq_length,
                VOCAB_SIZE_WORDS=len(self.token_model.vocab_list),
                VOCAB_SIZE_POS=len(self.token_model.pos_tags_list),
                INITIALIZER_TYPE=args.initializer_type,
                EMBEDDING_SIZE_WORDS=args.embedding_size,
                EMBEDDING_SIZE_POS=args.embedding_size,
                do_batch_norm=args.do_batchnorm,
                do_dropout=args.do_dropout,
                dropout_rate=args.dropout_rate,
                num_fc_layers=args.n_fc_layers,
                do_top_k=args.do_topk,
                train_word_embedding=False,
                train_pos_embedding=True,
                embedding_matrix=self.token_model.embedding_matrix,
                N_CHANNELS=args.n_channels
            )
        if self.args.model_type == "deep":
            logging.info("Compiling {} model".format(self.args.model_type))
            cnn_mdl = cnn_models.DeepWordPosCNN(
                ESSAY_LENGTH_WORDS=args.seq_length,
                VOCAB_SIZE_WORDS=len(self.token_model.vocab_list),
                VOCAB_SIZE_POS=len(self.token_model.pos_tags_list),
                INITIALIZER_TYPE=args.initializer_type,
                EMBEDDING_SIZE_WORDS=args.embedding_size,
                EMBEDDING_SIZE_POS=args.embedding_size,
                do_batch_norm=args.do_batchnorm,
                do_dropout=args.do_dropout,
                dropout_rate=args.dropout_rate,
                num_fc_layers=args.n_fc_layers,
                do_top_k=args.do_topk,
                train_word_embedding=False,
                train_pos_embedding=True,
                embedding_matrix=self.token_model.embedding_matrix,
                pos_embedding_matrix=self.pos_emb_matrix,
                N_CHANNELS=args.n_channels
            )
        return cnn_mdl

    def train_model(self, args):
        # get all the args variables
        experiment_prefix = args.experiment_name
        num_epochs = args.n_epochs
        logging.basicConfig(level=logging.INFO)

        self.x_train, self.x_test, self.y_train, self.y_test, \
        self.normalizer, self.token_model = data_loader.load_asap_data(
            data_path=args.data_path,
            set_no=args.set_no,
            embedding_path=args.embedding_path,
            seq_length=args.seq_length,
            embedding_size_words=args.embedding_size,
            onehot=args.model_type,
            train_all_sets=args.train_all_sets)

        self.x_holdout, self.y_holdout = data_loader.load_holdout_set(
            data_path=args.data_path,
            set_no=args.set_no,
            normalizer=self.normalizer,
            token_model=self.token_model,
            onehot=args.model_type)
        print("Starting training")
        time_stamp = str(int(time.time()))
        y_test_actuals = self.normalizer.inverse_transform(
            self.y_test).astype(
            int)

        y_holdout_actuals = self.normalizer.inverse_transform(
            self.y_holdout).astype(
            int)

        if args.self_trained_pos_embedding:
            with open('emb.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                self.pos_emb_matrix = pkl.load(f)
        else:
            self.pos_emb_matrix = None
        model_metrics = {"dev": {i: {} for i in range(args.n_runs)}, "holdout": {i: {} for i in range(args.n_runs)}}
        for run_number in range(args.n_runs):
            tf.reset_default_graph()
            cnn_mdl = self.get_cnn_model_object()
            prefix = "experiment_{}_{}_{}".format(args.experiment, args.set_no, args.variable)
            # Define the paths where we are going to store model information for
            # tensorboard
            logging.info("Running iteration {}".format(run_number))
            out_dir = "models_data"
            # These five lines creates directories to store stats
            model_stats_dir = os.path.join(out_dir, prefix, "model_stats")
            results_dir = os.path.join(out_dir, prefix, "results")
            checkpoint_dir = os.path.join(out_dir, prefix, "checkpoints")
            projector_dir = os.path.join(out_dir, prefix, "projector")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            train_summary_dir = os.path.join(out_dir, prefix, "summaries",
                                             "train")
            # If the model paths don't exist create them
            if not os.path.exists(projector_dir):
                os.makedirs(projector_dir)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if not os.path.exists(model_stats_dir):
                os.makedirs(model_stats_dir)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            # best_model = -1.0
            # Define an object to record custom stats
            # also record model stats and save as a json as it trains
            model_stats = {"train": {"loss": []},
                           "test": {"loss": []}}
            train_summary_writer = tf.summary.FileWriter(train_summary_dir)
            # defint the optimizar operation
            with tf.Session() as sess:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    print('learning rate:')
                    print(1 / (10 ** args.learning_power))
                    train_step = tf.train.AdamOptimizer(
                        1 / (10 ** args.learning_power)).minimize(cnn_mdl.loss)
                # global initializtion of variables
                sess.run(tf.global_variables_initializer())
                # add the current graph to the
                train_summary_writer.add_graph(sess.graph)
                # define the saver object for checkpoint files
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
                # Loop through the epochs
                ctr = 0
                best_model = 9999999999999999999999.99
                for epoch in range(num_epochs):
                    # define stats we want to see at the end of each epoch
                    tr_ctr = 0
                    tr_loss = 0.0
                    tr_acc = 0.0
                    logging.info("Epoch %d" % epoch)
                    logging.info("-" * 50)
                    # loop through all train batches
                    for batch in data_loader.iterate_minibatches(
                            self.x_train,
                            self.y_train,
                            16):
                        feed_dict = {cnn_mdl.input_x: batch[0],
                                     cnn_mdl.input_y: batch[1],
                                     cnn_mdl.phase: True}
                        # train the neural network for a single batch
                        _, l = sess.run([train_step, cnn_mdl.loss],
                                        feed_dict)
                        # accumulate train loss
                        tr_loss += l
                        tr_ctr += 1
                        ctr += 1
                    # Print stats while training
                    time_str = datetime.datetime.now().isoformat()
                    logging.info(
                        "Train {}: epoch {}, loss {:g}".format(time_str,
                                                               epoch,
                                                               tr_loss / tr_ctr))
                    # capture the epoch loss and accuracy
                    model_stats["train"]["loss"].append(tr_loss / tr_ctr)
                    # initialize test set stats for the epoch
                    te_ctr = 0
                    te_loss = 0.0
                    # loop through all test batches
                    for batch in data_loader.split_index(self.x_test,
                                                         self.y_test,
                                                         15):
                        feed_dict = {cnn_mdl.input_x: batch[0],
                                     cnn_mdl.input_y: batch[1],
                                     cnn_mdl.phase: False}
                        l = sess.run([cnn_mdl.loss], feed_dict)
                        te_loss += l[0]
                        te_ctr += 1
                    # capture test stats for the epoch
                    time_str = datetime.datetime.now().isoformat()
                    logging.info(
                        "Test {}: epoch {}, loss {:g}".format(time_str,
                                                              epoch,
                                                              te_loss / te_ctr))
                    model_stats["test"]["loss"].append(te_loss / te_ctr)
                    if (te_loss / te_ctr) < best_model:
                        best_model = (te_loss / te_ctr)
                        logging.info(
                            "New best model saving parameters {:.4f} ".format(
                                best_model))
                        saver.save(sess, checkpoint_prefix,
                                   global_step=epoch + 1)
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
                all_predictions = []
                te_ctr = 0
                graph = tf.get_default_graph()
                predictions = graph.get_operation_by_name(
                    "output/predictions/Sigmoid").outputs[0]
                if args.train_all_sets:
                    embeddings = graph.get_tensor_by_name("pos_embeddings_embedding_W1:0")
                for x_batch, y_batch in data_loader.split_index(
                        self.x_test,
                        self.y_test, 15):
                    feed_dict = {cnn_mdl.input_x: x_batch,
                                 cnn_mdl.input_y: y_batch,
                                 cnn_mdl.phase: False}
                    predictionslist = sess.run([predictions], feed_dict)
                    all_predictions.extend(predictionslist[0])
                    te_ctr += 1

                all_predictions_holdout = []
                te_ctr = 0
                for x_batch, y_batch in data_loader.split_index(self.x_holdout,
                                                                self.y_holdout,
                                                                15):
                    feed_dict = {cnn_mdl.input_x: x_batch,
                                 cnn_mdl.input_y: y_batch,
                                 cnn_mdl.phase: False}
                    predictionslist = sess.run([predictions],
                                               feed_dict)
                    all_predictions_holdout.extend(predictionslist[0])
                    te_ctr += 1
                final_predictions = np.around(
                    self.normalizer.inverse_transform(all_predictions)).astype(
                    int)
                recall = recall_score(y_test_actuals,
                                      final_predictions, average='macro')
                logging.info("Confusion Matrix")
                conf_matrix = confusion_matrix(y_test_actuals, final_predictions)
                acc_score = accuracy_score(y_test_actuals,
                                           final_predictions)  # float(te_acc / te_ctr)
                logging.info("Dev set accuracy score is {}".format(acc_score))
                qwk_score = cohen_kappa_score(y_test_actuals,
                                              final_predictions,
                                              weights="quadratic").tolist()
                logging.info("Dev set QWK score is {}".format(qwk_score))

                model_metrics["dev"][run_number]["recall"] = recall.tolist()
                model_metrics["dev"][run_number][
                    "predictions"] = final_predictions.tolist()
                model_metrics["dev"][run_number][
                    "actuals"] = y_test_actuals.tolist()
                model_metrics["dev"][run_number][
                    "accuracy"] = acc_score.tolist()
                model_metrics["dev"][run_number]["QWK"] = qwk_score
                model_metrics["dev"][run_number][
                    "confusion_matrix"] = conf_matrix.tolist()
                ##############################################################
                # compute holdout set metrics
                holdout_predictions = np.around(
                    self.normalizer.inverse_transform(
                        all_predictions_holdout)).astype(int)
                recall = recall_score(y_holdout_actuals,
                                      holdout_predictions, average=None)
                conf_matrix = confusion_matrix(y_holdout_actuals,
                                               holdout_predictions)
                acc_score = accuracy_score(y_holdout_actuals,
                                           holdout_predictions)  # float(te_acc / te_ctr)
                logging.info("Holdout set accuracy score is {}".format(acc_score))
                qwk_score = cohen_kappa_score(y_holdout_actuals,
                                              holdout_predictions,
                                              weights="quadratic").tolist()
                logging.info("Holdout set QWK score is {}".format(qwk_score))
                model_metrics["holdout"][run_number][
                    "recall"] = recall.tolist()
                model_metrics["holdout"][run_number][
                    "predictions"] = final_predictions.tolist()
                model_metrics["holdout"][run_number][
                    "actuals"] = y_holdout_actuals.tolist()
                model_metrics["holdout"][run_number][
                    "accuracy"] = acc_score.tolist()
                model_metrics["holdout"][run_number]["QWK"] = qwk_score
                model_metrics["holdout"][run_number][
                    "confusion_matrix"] = conf_matrix.tolist()
                if args.train_all_sets:
                    embedding = sess.run(embeddings)
            with open(os.path.join(model_stats_dir, "model_stats.json"),
                      "w") as w1:
                json.dump(model_stats, w1)
        with open(os.path.join(model_stats_dir, "args.pkl"), "wb") as w1:
            pkl.dump(args, w1)
        with open(os.path.join(model_stats_dir, "token_model.pkl"),
                  "wb") as w2:
            pkl.dump(self.token_model.vocab_list, w2)
        with open(os.path.join(results_dir, "prediction_summary.json"),
                  "w") as w:
            print('dumping prediction')
            json.dump(model_metrics, w)
        if args.train_all_sets:
            with open('emb.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pkl.dump(embedding, f)

    def predict_model(self, args, prefix, run_number=0):
        logging.basicConfig(level=logging.INFO)
        # We reset the graph initially and load it from disk later
        tf.reset_default_graph()
        if args.predict:
            self.x_train, self.x_test, self.y_train, self.y_test, \
            self.normalizer, self.token_model, self.x_train_tags, self.x_test_tags, self.enc = data_loader.load_asap_data(
                data_path=args.data_path, set_no=args.set_no,
                embedding_path=args.embedding_path, seq_length=args.seq_length,
                embedding_size_words=args.embedding_size)
        out_dir = "models_data"
        model_files = os.path.join(out_dir, prefix)
        meta_file = find_meta_file(model_files)
        tf_model = '%s/checkpoints/%s' % (model_files, meta_file)
        tf_checkpoints = '%s/checkpoints/' % model_files
        all_predictions = []
        results_directory = os.path.join(model_files, "results")

        logging.info("Making predictions for essay set {}".format(args.set_no))
        if not os.path.exists(results_directory):
            os.mkdir(results_directory)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(tf_model)
            saver.restore(sess, tf.train.latest_checkpoint(tf_checkpoints))
            # get the inpyt node from the graph
            graph = tf.get_default_graph()
            input_x = graph.get_operation_by_name("input_x").outputs[
                0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            # get the softmax layer from the graph
            predictions = graph.get_operation_by_name(
                "output/predictions/Sigmoid").outputs[0]
            # get the phase to input
            phase = graph.get_operation_by_name("phase").outputs[0]
            te_ctr = 0
            te_acc = 0.0
            for x_batch, y_batch in data_loader.split_index(
                    self.x_test,
                    self.y_test, 15):
                feed_dict = {input_x: x_batch,
                             input_y: y_batch,
                             phase: False}
                predictionslist = sess.run([predictions], feed_dict)
                all_predictions.extend(predictionslist[0])
                te_ctr += 1

            all_predictions_holdout = []
            te_ctr = 0
            for x_batch, y_batch in data_loader.split_index(self.x_holdout,
                                                            self.y_holdout,
                                                            15):
                feed_dict = {input_x: x_batch,
                             input_y: y_batch,
                             phase: False}
                predictionslist = sess.run([predictions], feed_dict)
                all_predictions_holdout.extend(predictionslist[0])
                te_ctr += 1

        with open(os.path.join(results_directory, "predictions.pkl"), \
                  "wb") as w:
            pkl.dump((all_predictions, final_predictions, self.y_test), w)

        with open(os.path.join(results_directory, "prediction_summary.json"),
                  "w") as w:
            json.dump(model_metrics, w)
        return final_predictions, self.y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument("--model-type",
                        type=str,
                        default=None,
                        choices=["words", "char", "both", "words-pos", "shallow", "deep", "deep-onehot"],
                        help="")
    parser.add_argument("--initializer-type",
                        type=str,
                        default="random-normal",
                        choices=["he-normal", "random-normal", "xavier"],
                        help="")
    parser.add_argument("--n-runs",
                        type=int,
                        default=1,
                        help="Number of runs")
    parser.add_argument("--n-epochs",
                        type=int,
                        default=50,
                        help="number of epochs")
    parser.add_argument("--train-overall",
                        action="store_true",
                        default=False,
                        help="train")
    parser.add_argument("--experiment-name",
                        type=str,
                        default="experiment_name",
                        help="")
    parser.add_argument("--data-path",
                        type=str,
                        default="data/training_data.csv",
                        help="")
    # neural network parameters
    parser.add_argument("--essay-length",
                        type=int,
                        default=1024,
                        help="")
    parser.add_argument("--vocab-size",
                        type=int,
                        default=71,
                        help="Size of char/word vocab")
    parser.add_argument("--n-scores",
                        type=int,
                        default=4,
                        help="Number of score points training on")
    parser.add_argument("--embedding-size",
                        type=int,
                        default=50,
                        help="embedding size")
    parser.add_argument("--load-pretrained",
                        action="store_true",
                        default=False,
                        help="train")
    parser.add_argument("--pretrained-model",
                        type=str,
                        default="some_directory",
                        help="")
    parser.add_argument("--do-dropout",
                        action="store_true",
                        default=False,
                        help="Use dropout regulerization")
    parser.add_argument("--do-batchnorm",
                        action="store_true",
                        default=False,
                        help="Use dropout regulerization")
    parser.add_argument("--dropout-rate",
                        type=float,
                        choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                 0.8, 0.9],
                        default=0.3,
                        help="use this value to try different dropout rates "
                             "as decimal values between 0 and 1")
    parser.add_argument("--cnn-model",
                        type=str,
                        choices=["cnn-1", "cnn-2", "cnn-vdcnn9",
                                 "cnn-baseline"],
                        default="cnn-1",
                        help="use this value to try different dropout rates "
                             "as decimal values between 0 and 1")
    parser.add_argument("--predict",
                        action="store_true",
                        default=False,
                        )
    parser.add_argument("--model-directory",
                        type=str,
                        default="some_directory",
                        help="")
    parser.add_argument("--n-fc-layers",
                        type=int,
                        default=2,
                        help="")
    parser.add_argument("--include-holdout",
                        action="store_true",
                        default=False,
                        help="Use dropout regulerization")
    parser.add_argument("--holdout-sample",
                        type=float,
                        choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                 0.8, 0.9],
                        default=0.3,
                        help="use this value to try different dropout rates "
                             "as decimal values between 0 and 1")
    parser.add_argument("--do-topk",
                        action="store_true",
                        default=False,
                        help="Use dropout regulerization")
    parser.add_argument("--normalize-predictions",
                        action="store_true",
                        default=False,
                        help="")
    parser.add_argument("--embedding-path",
                        type=str,
                        default="data/glove.6B.50d.txt",
                        help="Path of glove embeddings file")
    parser.add_argument("--set-no",
                        type=int,
                        default=1,
                        help="Essay set number")
    parser.add_argument("--seq-length",
                        type=int,
                        default=500,
                        help="Number of words in each sample")
    parser.add_argument("--n-filters",
                        type=int,
                        default=10,
                        help="Number of filters")
    parser.add_argument("--filter-size-1",
                        type=int,
                        default=2,
                        help="Filter size in 1st conv layer")
    parser.add_argument("--filter-size-2",
                        type=int,
                        default=3,
                        help="Filter size in 2nd conv layer")
    parser.add_argument("--filter-size-3",
                        type=int,
                        default=4,
                        help="Filter size in 3rd conv layer")
    parser.add_argument("--strides",
                        type=int,
                        default=3,
                        help="strides")
    parser.add_argument("--ksize",
                        type=int,
                        default=3,
                        help="ksize")
    parser.add_argument("--n-channels",
                        type=int,
                        default=2,
                        help="n-channels")
    parser.add_argument("--learning-power",
                        type=int,
                        default=3,
                        help="learning-power")
    parser.add_argument("--train-all-sets",
                        action="store_true",
                        default=False,
                        )
    parser.add_argument("--self-trained-pos-embedding",
                        action="store_true",
                        default=False,
                        )
    parser.add_argument("--experiment",
                        type=int,
                        default=1,
                        help='Experiment ID'
                        )
    parser.add_argument("--variable",
                        type=str,
                        default='',
                        help='Variable in experiment for identification'
                        )
    args = parser.parse_args()
    train_obj = TrainModel()
    if args.do_dropout and args.do_batchnorm:
        logging.error("Batchnorm and dropout can not be true at the same time")
    if args.set_no or args.train_all_sets:
        train_obj.train_model(args)
    elif args.predict:
        for i in range(0, args.n_runs):
            #             prefix = "set_" + str(args.set_no) + "_run_" + str(
            #                 i) + "_nf_" + str(args.n_filters) + "_fs1_" + str(
            #                 args.filter_size_1) + "_fs2_" + str(
            #                 args.filter_size_2) + "_fs3_" + str(
            #                 args.filter_size_3) + "_dp_" + str(
            #                 args.dropout_rate) + '_2fc_str_' + str(
            #                 args.strides) + '_ksize_' + str(args.ksize) + '_lp_' + str(
            #                 args.learning_power) + '_pos'
            prefix = "experiment_{}_{}_{}".format(args.experiment, args.set_no, args.variable)
            train_obj.predict_model(args, prefix)
