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

CKPT_DIR = 'models'


class Predict(object):

    def __init__(self):
        # 清除默认图的堆栈，并设置全局图为默认图
        # 若不进行清楚则在第二次加载的时候报错，因为相当于重新加载了两次
        self.config_path = "sentiment_analysis.config"
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path, encoding='utf-8-sig')

        test_batch_size = float(self.config.get("lstm_hyperparameter", "test_batch_size"))
        num_classes = int(self.config.get("lstm_hyperparameter", "num_classes"))
        self.data_helper = DataPreprocess()

        tf.reset_default_graph()

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        self.sess = tf.Session(config=session_conf)
        self.net = Network(batch_size=test_batch_size, num_class=num_classes)

        # 加载模型到sess中
        self.restore()
        print('load susess')

    def restore(self):
        saver = tf.train.Saver()
        checkpoint_file = tf.train.latest_checkpoint(CKPT_DIR)
        saver.restore(self.sess, checkpoint_file)

    def predict(self):
        while True:
            try:
                # 获取测试的句子向量
                input_sen = raw_input("input a sentence:\n")
                if input_sen == "end":
                    break
                input_sen_list = []

                input_sen_list.append(input_sen)
                input_sen_list = self.data_helper.do_tokenize(input_sen_list)
                sen_array = self.data_helper.build_train_sen(sen_list=input_sen_list, max_sen_length=28, model=model)
                print("length of input_sen_list", len(input_sen_list))

                print("batches shape ", sen_array.shape)

                y, predictions, scores = self.sess.run([self.net.y, self.net.predictions, self.net.scores],
                                                       feed_dict={self.net.input_data: sen_array})

                print(
                    ' Predict digit: y, predictions, scores: ', np.argmax(y[0]), predictions, scores)

            except Exception as e:
                print(e)


if __name__ == '__main__':
    model = Predict()
    model.predict()
