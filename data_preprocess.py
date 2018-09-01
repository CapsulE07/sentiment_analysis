# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import logging
import configparser
import numpy as np
import jieba
import os
import sys
from collections import Counter
import threadpool
from random import randint
import time

drop_stopwords_sentences_list = []
stopwords_list = []


class DataPreprocess(object):
    def __init__(self):
        config_path = "sentiment_analysis.config"
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8-sig')
        # self.word2vec_model = KeyedVectors.load_word2vec_format(self.config.get("input_file_path", "word2vec_bin_path"),
        #                                                         binary=True)

        self.labels_path = self.config.get("word2vec_parameter", "labels_path")
        self.total_comment_matrix_path = self.config.get("word2vec_parameter", "total_comment_matrix_path")

        # model, vector, word_list, neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length = self.load_file()

        # self.neg_content_len, self.pos_content_len, self.total_content_matrix, self.model, self.max_sen_length = self.build_sen2vec_matrix(
        #     model, neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length)

        # 生成向量text 和labels
        # total_content_matrix, labels = self.generatewordembeddings(
        #     model, neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length)

    def duplicate_remove(self, sentences_list):
        sentences_list = list(set(sentences_list))
        return sentences_list

    def tokenize(self, sentence_list):
        tokenized_sentences_list = []
        for sentence in sentence_list:
            one_sentence = []
            cut_sentence = jieba.cut(sentence)
            for word in cut_sentence:
                one_sentence.append(word)
            tokenized_sentences_list.append(one_sentence)
        return tokenized_sentences_list

    def threadpool_drop_stopwords(self, sentence):
        drop_stopwords_one_sentence = ""
        for word in sentence:
            if word not in stopwords_list:
                if word != "\t":
                    drop_stopwords_one_sentence += word
        drop_stopwords_sentences_list.append(drop_stopwords_one_sentence)

    def drop_stopwords(self, sentences_list):
        tokenized_sentences_list = self.tokenize(sentences_list)
        with open(self.config.get("input_file_path", "stopwords_path"), 'r') as stop_words_file:
            for stop_words in stop_words_file.readlines():
                stopwords_list.append(stop_words.replace("\n", ""))
        pool = threadpool.ThreadPool(20)
        requests = threadpool.makeRequests(self.threadpool_drop_stopwords, tokenized_sentences_list)
        [pool.putRequest(req) for req in requests]
        pool.wait()
        return drop_stopwords_sentences_list

    def data_preprocess(self, sentences_list, remove_duplicate=True, stopwords_drop=False):
        if remove_duplicate:
            sentences_list = self.duplicate_remove(sentences_list)

        if stopwords_drop:
            self.drop_stopwords(sentences_list)
        return sentences_list

    """
    read negative and positive corpus in corpus folder, and collect these corpus in one list dividually. 

    Args: 
        file_path = negative or positive corpus foleder path
    Return:
        file_content_list which includes negative or positive all corpus		
    """

    def read_file_name(self, file_dir):
        file_content_list = []
        for root, dirs, files in os.walk(file_dir):
            for file_name in files:
                with open(os.path.join(root, file_name), 'r') as corpus_file:
                    file_content_list = corpus_file.readlines()
        return file_content_list

    """
    1. this function decides the max length of sentences we need to select
    2. we use the 'most_common' function to adjust how many common sentences ought to be considered, where most_common ratio can be set in a config file.
    3. we will select the max length of sentence which are in common sentences as the max_sen_length  
    arg:
        tokenized_content_list: these sentences have been tokenized
    return:
        max_sen_length: this is a 'int' format
    """

    def calculate_len_sen(self, tokenized_content_list):
        sen_len_list = []
        print("tokenized_content_list 的长度为 {0}".format(len(tokenized_content_list)))
        for sen in tokenized_content_list:
            sen_len_list.append(len(sen))
        sen_counter = Counter(sen_len_list)
        most_sen_tuple = sen_counter.most_common(
            int(float(self.config.get("threshold_parameter", "most_percentage")) * len(sen_len_list)))
        len_collection_list = []
        for len_tuple in most_sen_tuple:
            len_collection_list.append(len_tuple[0])
        max_sen_length = max(len_collection_list)
        return max_sen_length

    """
    this function is used for tokenizing sentences with jieba tool
    for example:
    input = ["今天天气不错","上海今天温度是多少度？"]
    output = [["今天","天气","不错"],["上海","今天","温度","是","多少","度","？"]]

    args：
        sen_list: these sentences is used to be tokenized
    return:
        tokenized_content_list: these sentences have been tokenized
    """

    def do_tokenize(self, sen_list):
        tokenized_content_list = []
        for sen in sen_list:
            sen_list = []
            cut_sen = jieba.cut(sen)
            for i in cut_sen:
                sen_list.append(i)
            tokenized_content_list.append(sen_list)
        return tokenized_content_list

    """
    1. chiefly this function is aim to load file, where detail steps are shown belown:
        a. load the word2vec bin model
        b. load the word2vec npy model
        c. load the negative/positive comment text file
    2. then the loaded comment files will be preprocessed in data_preprocess
        a. remove_duplicate argument call help users to remove duplicate comment
        b. stopwords_drop argument can help users to remove stopwords
    3. the last step is tokenizing these sentences and calculate the max length of sentences	

    args:
        null
    return:
        neg_tokenized_content_list: negative content which have tokenized list
        pos_tokenized_content_lsit: positive content which have tokenized list
        max_sen_length: the max_length of sentences that we need to select
    """


    def load_word2vec(self):
        start = time.time()
        word2vec_model = KeyedVectors.load_word2vec_format(self.config.get("input_file_path", "word2vec_bin_path"),
                                                           binary=True)
        tmp = time.time()
        print("加载 word2vec_model 花费时间为： ", tmp - start)
        logging.info("loading word2vec bin file...")
        logging.info("loading word2vec npy file ...")
        vector = np.load(self.config.get("input_file_path", "word2vec_npy_path"))
        word_list = word2vec_model.wv.vocab.keys()

        end = time.time()
        print("加载vector ，word_list花费时间为： " , end - tmp)
        return word2vec_model, vector, word_list

    def load_corpus(self):
        logging.info("loading nagetive and positive file ...")
        neg_content_list = self.read_file_name(self.config.get("input_file_path", "neg_file_path"))
        pos_path = self.config.get("input_file_path", "pos_file_path")
        print("pos_content_list 路径为： ", pos_path)
        pos_content_list = self.read_file_name(pos_path)
        print("neg_content_list 的长度为 {0}".format(len(neg_content_list)))
        print("pos_content_list 的长度为 {0}".format(len(pos_content_list)))
        logging.info("data preprocessing ...")

        neg_content_list = self.data_preprocess(sentences_list=neg_content_list, remove_duplicate=True,
                                                stopwords_drop=False)
        pos_content_list = self.data_preprocess(sentences_list=pos_content_list, remove_duplicate=True,
                                                stopwords_drop=False)
        print("neg_content_list 的长度为 {0}".format(len(neg_content_list)))
        print("pos_content_list 的长度为 {0}".format(len(pos_content_list)))
        logging.info("sentence tokenized ...")
        neg_tokenized_content_list = self.do_tokenize(neg_content_list)
        pos_tokenized_content_list = self.do_tokenize(pos_content_list)
        max_sen_length = self.calculate_len_sen(pos_tokenized_content_list + neg_tokenized_content_list)

        neg_content_len = len(neg_tokenized_content_list)
        pos_content_len = len(pos_tokenized_content_list)
        return neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length, neg_content_len, pos_content_len

    def build_train_sen(self, sen_list, max_sen_length, model):
        total_sen_vec_list = []
        word_dimension = int(self.config.get("word2vec_parameter", "dimension"))
        for record_num, sen in enumerate(sen_list):
            one_sen_vec_list = []
            if len(sen) >= max_sen_length:
                for num in range(0, max_sen_length):
                    try:
                        one_sen_vec_list.append(model[sen[num]])
                    except:
                        one_sen_vec_list.append(np.zeros(word_dimension))
            else:
                for word in sen:
                    try:
                        one_sen_vec_list.append(model[word])
                    except:
                        one_sen_vec_list.append(np.zeros(word_dimension))
                for num in range(len(sen) + 1, max_sen_length + 1):
                    one_sen_vec_list.append(np.zeros(word_dimension))
            total_sen_vec_list.append(np.array(one_sen_vec_list))
        return np.array(total_sen_vec_list)

    def get_train_batch(self, total_content_matrix, neg_content_len, pos_content_len, max_sen_length):
        batch_size = int(self.config.get("lstm_hyperparameter", "batch_size"))
        word_dimension = int(self.config.get("word2vec_parameter", "dimension"))
        labels = []
        batch_matrix = np.zeros([batch_size, max_sen_length, word_dimension])
        for i in range(batch_size):
            if (i % 2 == 0):
                num = randint(1, neg_content_len - 1000)
                labels.append([0, 1])
            else:
                num = randint(neg_content_len + 2, neg_content_len + pos_content_len - 1000)
                labels.append([1, 0])
            batch_matrix[i] = total_content_matrix[num - 1:num]
        return batch_matrix, labels

    def get_test_batch(self, total_content_matrix, neg_content_len, pos_content_len, max_sen_length):
        batch_size = int(self.config.get("lstm_hyperparameter", "batch_size"))
        word_dimension = int(self.config.get("word2vec_parameter", "dimension"))
        labels = []
        batch_matrix = np.zeros([batch_size, max_sen_length, word_dimension])
        for i in range(batch_size):
            if (i % 2 == 0):
                num = randint(neg_content_len - 1999, neg_content_len + 1)
                labels.append([0, 1])
            else:
                num = randint(neg_content_len + pos_content_len - 1999, neg_content_len + pos_content_len)
                labels.append([1, 0])
            batch_matrix[i] = total_content_matrix[num - 1:num]
        return batch_matrix, labels

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def generatewordembeddings(self, model, neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length,
                               ):
        neg_content_len = len(neg_tokenized_content_list)
        pos_content_len = len(pos_tokenized_content_list)
        neg_content_array = self.build_train_sen(neg_tokenized_content_list, max_sen_length, model)
        pos_content_array = self.build_train_sen(pos_tokenized_content_list, max_sen_length, model)
        total_content_matrix = np.concatenate((neg_content_array, pos_content_array), axis=0)
        labels = np.ones((len(total_content_matrix), 1))
        labels[:neg_content_len] = 0

        if not os.path.isfile(self.total_comment_matrix_path):
            file = open(self.total_comment_matrix_path, 'w')
            file.close()
        if not os.path.isfile(self.labels_path):
            file = open(self.labels_path, 'w')
            file.close()

        """
        import numpy as np
        保存多维数组的方式
        a = np.random.random((2, 3, 4, 5))
        header = ','.join(map(str, a.shape))
        np.savetxt('test.txt', a.reshape(-1, a.shape[-1]), header=header,
           delimiter=',')
        加载保存多维数组的方式
         with open('test.txt') as f:
            shape = map(int, f.next()[1:].split(','))
            b = np.genfromtxt(f, delimiter=',').reshape(shape)  
            
        # 在python2时， 使用 f.next()
        # python3
        #map(function, iterable, ...)  # 返回的不是list 而是 map object
        #list(map(function, iterable, ...))  # 转成list
        """
        total_content_matrix_header = ','.join(map(str, total_content_matrix.shape))
        np.savetxt(self.total_comment_matrix_path, total_content_matrix.reshape(-1, total_content_matrix.shape[-1]),
                   header=total_content_matrix_header,
                   delimiter=',')

        labels_header = ','.join(map(str, labels.shape))
        np.savetxt(self.labels_path, labels.reshape(-1, labels.shape[-1]), header=labels_header,
                   delimiter=',')

        # print("total_content_matrix_header: " + total_content_matrix)
        # print("labels_header: " + labels_header)

        return total_content_matrix, labels


    def loadembeddings(self):
        with open(self.total_comment_matrix_path) as f:
            shape = list(map(int, f.readline()[1:].split(',')))
            total_content_matrix = np.genfromtxt(f, delimiter=',').reshape(shape)

        with open(self.labels_path) as f:
            shape = list(map(int, f.readline()[1:].split(',')))
            labels = np.genfromtxt(f, delimiter=',').reshape(shape)
        return total_content_matrix, labels


if __name__ == '__main__':
    data = DataPreprocess()
    # 加载语料库
    neg_tokenized_content_list, pos_tokenized_content_list, max_sen_length, neg_content_len, pos_content_len = data.load_corpus()
    #加载Word2vec 模型
    word2vec_model, vector, word_list = data.load_word2vec()
    # 生成 评论向量 矩阵
    total_content_matrix, labels = data.generatewordembeddings(word2vec_model, neg_tokenized_content_list,
                                                               pos_tokenized_content_list, max_sen_length)
    # 加载 评论向量 矩阵
    total_content_matrix, labels = data.loadembeddings()
    print(total_content_matrix.shape)
    print(labels.shape)
