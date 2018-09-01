# -*- coding: utf-8 -*-

import tensorflow as tf
import logging
import configparser
import os
from data_preprocess import DataPreprocess
import sys
from network import Network
import numpy as np

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
os.environ["CUDA_VISIBLE_DEVICES"] = ""
CKPT_DIR = "model"


class Evaluate(object):
    def __init__(self):
        self.data_helper = DataPreprocess()
        # 清除默认图的堆栈，并设置全局图为默认图
        # 若不进行清楚则在第二次加载的时候报错，因为相当于重新加载了两次
        self.config_path = "sentiment_analysis.config"
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path, encoding='utf-8-sig')

        self.total_content_matrix, self.neg_content_len, self.pos_content_len, self.max_sen_length = self.load_relevant_data()


        self.graph, self.sess = self.restore()
        logging.info('load success')

    def restore(self):
        # tf.reset_default_graph()
        checkpoint_file = tf.train.latest_checkpoint("models")
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                )
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
        return graph, sess

    def load_relevant_data(self):
        neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length, neg_content_len, pos_content_len = self.data_helper.load_corpus()
        total_content_matrix, labels = self.data_helper.loadembeddings()
        return total_content_matrix, neg_content_len, pos_content_len, max_sen_length


    def evaluate(self):
        # Get the placeholders from the graph by name
        input_data = self.graph.get_operation_by_name("input/input_data").outputs[0]
        labels = self.graph.get_operation_by_name("input/labels").outputs[0]
        # Tensors we want to evaluate
        final_predictions = self.graph.get_operation_by_name("accuracy/final_predictions").outputs[0]

        total_nums = 0
        total_true = 0

        for i in range(1000):
            next_batch, next_batch_labels = self.data_helper.get_test_batch(
                self.total_content_matrix, self.neg_content_len,
                self.pos_content_len, self.max_sen_length)
            batch_predictions = self.sess.run(final_predictions,
                                              feed_dict={input_data: next_batch, labels: next_batch_labels})

            print("the accuracy of batch {0} ： {1}".format(i + 1, sum(batch_predictions) / len(next_batch_labels)))

            total_nums += len(next_batch_labels)
            total_true += sum(batch_predictions)

        result = total_true / total_nums
        print("Total number of test examples: {0}".format(total_nums))
        print("Total accuracy of test examples: {0}".format(result))


if __name__ == '__main__':
    eva = Evaluate()
    eva.evaluate()
