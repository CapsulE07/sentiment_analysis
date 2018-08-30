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

        self.test_batch_size = float(self.config.get("lstm_hyperparameter", "test_batch_size"))
        self.num_classes = int(self.config.get("lstm_hyperparameter", "num_classes"))
        self.data_helper = DataPreprocess()

        # 加载模型到sess中
        self.graph, self.sess = self.restore()

    def restore(self):
        # tf.reset_default_graph()
        checkpoint_file = tf.train.latest_checkpoint(CKPT_DIR)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # set a new  meta graph and restore variables
                self.net = Network(batch_size=self.test_batch_size, num_class=self.num_classes)
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_file)
        print('load success')
        return graph, sess

    def predict(self, pre_string=None):
        """
            在有输入字符的情况下，返回测试结果
            在没有输入字符的情况下，无限循环，持续测试，直至输入的字符为"end"
        :rtype: str
        """
        # Get the placeholders from the graph by name
        input_data = self.graph.get_operation_by_name("input/input_data").outputs[0]
        labels = self.graph.get_operation_by_name("input/labels").outputs[0]
        # Tensors we want to evaluate
        predicitons = self.graph.get_operation_by_name("output/predictions").outputs[0]

        while True:
            try:
                # 获取测试的句子向量
                if pre_string:
                    input_sen = pre_string
                else:
                    input_sen = raw_input("input a sentence:\n")
                    if input_sen == "end":
                        break
                print("Predict sentence is: {0}".format(input_sen))

                input_sen_list = []
                input_sen_list.append(input_sen)
                input_sen_list = self.data_helper.do_tokenize(input_sen_list)

                sen_array = self.data_helper.build_train_sen(sen_list=input_sen_list, max_sen_length=28,
                                                             model=self.data_helper.model)

                result = self.sess.run([predicitons], feed_dict={input_data: sen_array})

                if int(result[0][0]) == 0:
                    res = "positive"
                else:
                    res = "negative"
                print(' Predict class is ： {0} '.format(res))
                # print(' Predict class is {0}: '.format(s))

                if pre_string:
                    return res

            except Exception as e:
                print(e)


if __name__ == '__main__':
    pre = Predict()
    pre.predict()
