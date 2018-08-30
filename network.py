# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import tensorflow as tf
import logging
import configparser
import numpy as np
import jieba
import os
from data_preprocess import DataPreprocess
import sys
from collections import Counter
import threadpool
from random import randint

reload(sys)
sys.setdefaultencoding('utf-8')

# setting logging configuration
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('log.log', mode='w', encoding='UTF-8')
fileHandler.setLevel(logging.NOTSET)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)


# 构建图
class Network(object):

    def __init__(self, batch_size, num_class):
        config_path = "sentiment_analysis.config"
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8-sig')

        batch_size = batch_size
        num_classes = num_class

        lstm_units = int(self.config.get("lstm_hyperparameter", "lstm_units"))
        output_keep_prob = float(self.config.get("lstm_hyperparameter", "output_keep_prob"))
        word_dimension = int(self.config.get("word2vec_parameter", "dimension"))
        learning_rate = float(self.config.get("lstm_hyperparameter", "learning_rate"))
        layers_num = int(self.config.get("lstm_hyperparameter", "layers_num"))
        l2_regularizer = float(self.config.get("lstm_hyperparameter", "l2_regularizer"))

        data_helper = DataPreprocess()

        ## 获取训练数据
        model, vector, word_list, neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length = data_helper.load_file()

        # """reset graph"""
        # tf.reset_default_graph()

        """set place holder"""
        with tf.variable_scope('input'):
            self.labels = tf.placeholder(dtype=tf.int64, shape=[batch_size, num_classes], name='labels')
            self.input_data = tf.placeholder(dtype=tf.float64, shape=[batch_size, max_sen_length, word_dimension],
                                             name='input_data')

        """forward propogation"""
        with tf.variable_scope('lstmCell'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
            multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_units) for _ in range(layers_num)])
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
            multi_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=multi_lstm_cell, output_keep_prob=output_keep_prob)
            outputs, _ = tf.nn.dynamic_rnn(multi_lstm_cell, self.input_data, dtype=tf.float64)

        """output port"""
        with tf.variable_scope("output"):
            weight = tf.get_variable(name="weight", shape=[lstm_units, num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
            # weight = tf.get_variable(name="weight", shape=[lstm_units, num_classes], initializer=tf.truncated_normal_initializer(stddev=0.1, seed = 1), dtype=tf.float64)
            bias = tf.get_variable(name="bias", shape=[num_classes], initializer=tf.constant_initializer(0.1),
                                   dtype=tf.float64)
            output = outputs[:, -1, :]

            self.scores = tf.nn.xw_plus_b(output, weight, bias, name="scores")
            self.y = tf.nn.softmax(self.scores,name="predict_classes")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Accuracy
        with tf.variable_scope("accuracy"):
            final_predictions = tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), dtype="float",
                                          name="final_predictions")
            self.accuracy = tf.reduce_mean(final_predictions, name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

        """loss function"""
        with tf.variable_scope("loss"):
            # l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_regularizer), tf.trainable_variables())
            l2_loss = tf.contrib.layers.l2_regularizer(l2_regularizer)(weight)
            logit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels))
            self.loss = logit_loss + l2_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            tf.summary.scalar("loss", self.loss)

        """session"""
        self.merged = tf.summary.merge_all()
