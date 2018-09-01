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
from network import Network
from eval import Evaluate


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

CKPT_DIR = "model"


# 构建图
class Train(object):
    def __init__(self):
        self.config_path = "sentiment_analysis.config"
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path, encoding='utf-8-sig')
        self.train_batch_size = int(self.config.get("lstm_hyperparameter", "batch_size"))

        self.data_helper = DataPreprocess()
        self.total_content_matrix, self.neg_content_len, self.pos_content_len, self.max_sen_length = self.load_relevant_data()

        # self.graph, self.sess = self.loadmodel()
        self.net = Network(batch_size=self.train_batch_size, num_class=2, max_sen_length=self.max_sen_length)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        self.sess = tf.Session(config=session_conf)
        self.sess.run(tf.global_variables_initializer())

    def load_relevant_data(self):
        neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length, neg_content_len, pos_content_len = self.data_helper.load_corpus()
        total_content_matrix, labels = self.data_helper.loadembeddings()
        return total_content_matrix, neg_content_len, pos_content_len, max_sen_length

    # def loadmodel(self):
    #     # tf.reset_default_graph()
    #     graph = tf.Graph()
    #     with graph.as_default():
    #         session_conf = tf.ConfigProto(
    #             allow_soft_placement=True,
    #             log_device_placement=True)
    #         sess = tf.Session(config=session_conf)
    #         with sess.as_default():
    #             # set a new  meta graph and restore variables
    #             self.net = Network(batch_size=self.train_batch_size, num_class=2, max_sen_length=self.max_sen_length)
    #             sess.run(tf.global_variables_initializer())
    #     print('load success')
    #     return graph, sess

    def train(self):

        iterations = int(self.config.get("lstm_hyperparameter", "iteration"))
        # tf.train.Saver用于保存训练的结果
        # max to keep 用于设置最多保存多少个模型
        # 如果保存的模型超过这个值，最旧的模型被删除
        saver = tf.train.Saver(max_to_keep=10)

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.get_checkpoint_state(CKPT_DIR):
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            # 读取网络中的global_step的值，即当前已经训练的次数
            step = self.sess.run(self.net.global_step)
            print('continue from')
            print('  -> Minibatch update : ', step)
        else:
            step = 0
        summary_writer = tf.summary.FileWriter("log", tf.get_default_graph())

        while step < iterations:

            next_batch, next_batch_labels = self.data_helper.get_train_batch(self.total_content_matrix,
                                                                             self.neg_content_len, self.pos_content_len,
                                                                             self.max_sen_length)
            summary, _ = self.sess.run([self.net.merged, self.net.optimizer],
                                       feed_dict={self.net.input_data: next_batch, self.net.labels: next_batch_labels})
            summary_writer.add_summary(summary, step)
            if (step % 200 == 0 and step != 0):
                loss_ = self.sess.run(self.net.loss,
                                      feed_dict={self.net.input_data: next_batch, self.net.labels: next_batch_labels})
                accuracy_ = self.sess.run(self.net.accuracy, feed_dict={self.net.input_data: next_batch,
                                                                        self.net.labels: next_batch_labels})
                print("iteration {}/{}   ".format(step + 1, iterations),
                      "loss{}   ".format(loss_),
                      "accuracy{}   ".format(accuracy_))
            if (step % 2000 == 0 and step != 0):
                save_path = saver.save(self.sess, self.config.get("output_file_path", "model"), global_step=step)
                print("saved to %s" % save_path)
            step += 1

        summary_writer.close()


if __name__ == '__main__':
    tra = Train()
    tra.train()
    # eval = Evaluate()
    # eval.evaluate()
