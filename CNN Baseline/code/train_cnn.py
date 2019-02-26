#!/usr/bin/env python

import argparse
from dataloader import data_loader
import tensorflow as tf
import cnn_models
import logging
import numpy as np
import datetime
import time
import os
import _pickle as pkl
import json

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

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
        self.x_train, self.x_test, self.y_train, self.y_test, self.token_model = None, None, None, None, None
    
    
    def train_model(self, args):
        # get all the args variables
        experiment_prefix = args.experiment_name
        num_epochs = args.n_epochs
        logging.basicConfig(level=logging.INFO)
        # def load_data(data_set, csv_path, prompt, trait, return_meta, model_type="words", add_holdout_examples=False):
        self.x_train, self.x_test, self.y_train, self.y_test, self.token_model, self.normalizer = data_loader.load_asap_data(data_path=args.data_path,set_no=args.set_no,embedding_path=args.embedding_path,seq_length=args.seq_length)
        n_scores = self.y_test.shape[1]
        logging.info("Training on trait {}".format(args.trait))
        logging.info("Modeling with {} scores".format(n_scores))
        if args.model_type == "words":
            if args.cnn_model == "cnn-baseline":
                cnn_mdl = cnn_models.CNN_baseline(
                      ESSAY_LENGTH_WORDS=args.seq_length,
                      VOCAB_SIZE_WORDS=len(self.token_model.vocab_list),
                      EMBEDDING_SIZE_WORDS=self.token_model.embedding_matrix.shape[1],
                      do_batch_norm=args.do_batchnorm,
                      do_dropout=args.do_dropout,
                      dropout_rate=args.dropout_rate,
                      num_fc_layers=args.n_fc_layers,
                      do_top_k=args.do_topk,
                      train_embedding=False,
                      embedding_matrix=self.token_model.embedding_matrix,
                    normalizer=self.normalizer
                )
        for i in range(args.n_runs):
            # record the timestep of every model
            #timestamp = experiment_prefix + str(int(time.time()))
            prefix = "set_" + str(args.set_no) + "_run_" + str(i)
            # Define the paths where we are going to store model information for
            # tensorboard
            out_dir = "models"
            # These five lines creates directories to store stats
            model_stats_dir = os.path.join(out_dir, prefix, "model_stats")
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

            #best_model = -1.0
            # Define an object to record custom stats
            # also record model stats and save as a json as it trains
            model_stats = {"train": {"loss": []},
                           "test": {"loss": []}}

            all_summaries = tf.summary.merge_all()
            train_summary_writer = tf.summary.FileWriter(train_summary_dir)
            # defint the optimizar operation
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # The following statement is there so that batch norm
            # can use population mean and standard deviations during test time
            # REQ: very important to be there the "control_dependencies"
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-3).minimize(cnn_mdl.loss)
            with tf.Session() as sess:
                # global initializtion of variables
                sess.run(tf.global_variables_initializer())
                # add the current graph to the
                train_summary_writer.add_graph(sess.graph)
                # define the saver object for checkpoint files
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
                # Loop through the epochs
                ctr = 0
                if args.model_type == "words":
                    for epoch in range(num_epochs):
                        # define stats we want to see at the end of each epoch
                        tr_ctr = 0
                        tr_loss = 0.0
                        tr_acc = 0.0
                        logging.info("Epoch %d" % epoch)
                        logging.info("-" * 50)
                        print("input x shape")
                        print(cnn_mdl.input_x.shape)
                        print(self.x_train.shape)
                        print(self.x_test.shape)
                        # loop through all train batches
                        for batch in data_loader.iterate_minibatches(self.x_train,
                                                                     self.y_train,
                                                                     15):
                            feed_dict = {cnn_mdl.input_x: batch[0],
                                         cnn_mdl.input_y: batch[1],
                                         cnn_mdl.phase: True}
                            # train the neural network for a single batch
                            _, l = sess.run([train_step,cnn_mdl.loss],
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
                        # save the model parameters for the best performing model on the
                        # test set
                        if epoch == 0:
                            best_model = (te_loss / te_ctr)
                            logging.info(
                                "New best model saving parameters {:.4f} ".format(
                                    best_model))
                            saver.save(sess, checkpoint_prefix, global_step=epoch + 1)
                        elif (te_loss / te_ctr) < best_model:
                            best_model = (te_loss / te_ctr)
                            logging.info(
                                "New best model saving parameters {:.4f} ".format(
                                    best_model))
                            saver.save(sess, checkpoint_prefix, global_step=epoch + 1)
            with open(os.path.join(model_stats_dir, "model_stats.json"), "w") as w1:
                json.dump(model_stats, w1)
            with open(os.path.join(model_stats_dir, "args.pkl"), "wb") as w1:
                pkl.dump(args, w1)
            with open(os.path.join(model_stats_dir, "token_model.pkl"), "wb") as w2:
                pkl.dump(self.token_model.vocab_list, w2)
            #args.model_directory = os.path.join(out_dir, timestamp)
            logging.info("Evaluating test set")
            if args.n_runs == 1:
                self.predict_model(args,prefix)

    def predict_model(self, args,prefix):
        logging.basicConfig(level=logging.INFO)
        #score_points = [0, 1, 2, 3, 4, 5]
        tf.reset_default_graph()
        # If we are not predicting then the train and test sets are loaded
        if args.predict:
            self.x_train, self.x_test, self.y_train, self.y_test, self.token_model, self.normalizer = data_loader.load_asap_data(data_path=args.data_path,set_no=args.set_no,embedding_path=args.embedding_path,seq_length=args.seq_length)
        out_dir = "models"
        model_files = os.path.join(out_dir, prefix)
        #model_files = args.model_directory
        meta_file = find_meta_file(model_files)
        tf_model = '%s/checkpoints/%s' % (model_files, meta_file)
        tf_checkpoints = '%s/checkpoints/' % model_files
        all_predictions = []
        results_directory = os.path.join(model_files, "results")
        
        logging.info("Making predictions for prompt {}".format(args.prompt))
        if not os.path.exists(results_directory):
            os.mkdir(results_directory)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(tf_model)
            saver.restore(sess, tf.train.latest_checkpoint(tf_checkpoints))
            # get the inpyt node from the graph
            graph = tf.get_default_graph()
            if args.model_type == "words" or args.model_type == "char":
                input_x = graph.get_operation_by_name("input_x_words").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                # get the softmax layer from the graph
                predictions = graph.get_operation_by_name("output/predictions/Sigmoid").outputs[0]
                #model_acc = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
                # get the phase to input
                phase = graph.get_operation_by_name("phase").outputs[0]
                te_ctr = 0
                te_acc = 0.0
                for x_batch, y_batch in data_loader.split_index(self.x_test,
                                                                self.y_test,15):
                    feed_dict = {input_x: x_batch,
                                 input_y: y_batch,
                                 phase: False}
                    predictionslist = sess.run([predictions], feed_dict)
                    all_predictions.extend(predictionslist[0])
                    te_ctr += 1
                print(self.y_test[0:10])
                print(all_predictions[0:10])
        #all_raw_predictions = np.concatenate(all_predictions)
#         if args.normalize_predictions:
#             score_distribution = np.bincount(np.argmax(self.y_train, axis = 1))
#             inverse_score_distribution = 1. / score_distribution
#             all_raw_predictions = inverse_score_distribution * all_raw_predictions
#             logging.info(all_raw_predictions.shape)
#         logging.info("Shape of predictions")
#         logging.info(all_predictions.shape)
        model_metrics = {}
        #self.y_test = np.argmax(self.y_test, axis=1)
        #final_predictions = np.argmax(all_raw_predictions, axis=0)
        final_predictions = all_predictions
        #self.y_test = self.y_test[:final_predictions.shape[0]]
        self.y_test = self.normalizer.inverse_transform(self.y_test).astype(int)
        final_predictions = np.around(self.normalizer.inverse_transform(final_predictions)).astype(int)
#         precision = precision_score(self.y_test,
#                                     final_predictions)
        recall = recall_score(self.y_test,
                              final_predictions, average='macro')
#         f1_scores = f1_score(self.y_test,
#                              final_predictions)
        # conf_matrix, d_frame = compute_conf_matrix_with_metrics(self.y_test, all_predictions, which_code_order, True)
        # plot_confusion_matrix(conf_matrix, which_code_order, results_directory + os.sep)
        # d_frame.to_csv(os.path.join(results_directory, "confusion_matrix.csv"))
#         class_report = classification_report(self.y_test,
#                                              final_predictions)
        #logging.info(class_report)
        #logging.info("-" * 50)
        logging.info("Confusion Matrix")
        conf_matrix = confusion_matrix(self.y_test, final_predictions)
        logging.info(pd.DataFrame(conf_matrix))
        acc_score = accuracy_score(self.y_test, final_predictions) #float(te_acc / te_ctr)
        logging.info("Final accuracy score is {}".format(acc_score))
#         prefix = ""
#         if args.prompt:
#             prefix = args.prompt
#         with open(os.path.join(results_directory,
#                                prefix + "classification_report.txt"), "wb") as f1:
#             f1.write(class_report)
        #model_metrics["precision"] = list(precision)
        model_metrics["recall"] = recall.tolist()
        #model_metrics["f1"] = list(f1_scores)
        model_metrics["predictions"] = final_predictions.tolist()
        model_metrics["actuals"] = self.y_test.tolist()
        model_metrics["accuracy"] = acc_score.tolist()
        model_metrics["QWK"] = cohen_kappa_score(self.y_test, final_predictions,
                                                 weights="quadratic").tolist()
        model_metrics["confusion_matrix"] = conf_matrix.tolist()
        print(model_metrics)
        with open(os.path.join(results_directory,"predictions.pkl"), \
                "wb") as w:
            pkl.dump((all_predictions, final_predictions, self.y_test), w)

        with open(os.path.join(results_directory, "prediction_summary.json"), "w") as w:
            json.dump(model_metrics, w)

        return final_predictions, self.y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument("--data-set",
                        type=str,
                        default=None,
                        choices=["aspire-math", "massachusetts", "parcc-math-text"],
                        help="")
    parser.add_argument("--model-type",
                        type=str,
                        default=None,
                        choices=["words", "char", "both"],
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
    parser.add_argument("--train-prompt",
                        action="store_true",
                        default=False,
                        help="train")
    parser.add_argument("--prompt",
                        type=str,
                        default=None,
                        help="")
    parser.add_argument("--experiment-name",
                        type=str,
                        default="experiment_name",
                        help="")
    parser.add_argument("--data-path",
                        type=str,
                        default="aspire-spring-2018-human-data-train-test-holdout-labels.csv",
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
                        default=16,
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
                        choices=["cnn-1", "cnn-2", "cnn-vdcnn9","cnn-baseline"],
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
    parser.add_argument("--trait",
                        type=str,
                        choices=["0", "1", "2", "A", "B", "", "default-trait", "a", "b"],
                        default="1",
                        help="use this value to try different dropout rates "
                             "as decimal values between 0 and 1")
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
                        default="glove.6B.50d.txt",
                        help="Path of glove embeddings file")
    parser.add_argument("--set-no",
                    type=int,
                    default=1,
                    help="Essay set number")
    parser.add_argument("--seq-length",
                    type=int,
                    default=500,
                    help="Number of words in each sample")
    args = parser.parse_args()
    train_obj = TrainModel()
    if args.do_dropout and args.do_batchnorm:
        logging.error("Batchnorm and dropout can not be true at the same time")
    if args.train_overall:
        train_obj.train_model(args)
    elif args.train_prompt:
        train_obj.train_model(args)
    elif args.predict:
        for i in range(0,args.n_runs):
            prefix = "set_" + str(args.set_no) + "_run_" + str(i)
            train_obj.predict_model(args,prefix)
