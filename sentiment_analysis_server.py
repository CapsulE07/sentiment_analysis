# -*- coding: utf-8 -*-

import flask
import json
import numpy as np
import urllib
import urllib2
from predict import Predict
import configparser
import time

app = flask.Flask(__name__)
config_path = "sentiment_analysis.config"
cf = configparser.ConfigParser()
cf.read(config_path)

FLASK_HOST = cf.get('sentiment_analysis_server', 'flask_host')
FLASK_PORT = cf.get('sentiment_analysis_server', 'flask_port')

GET_PREDICT_STRING = cf.get(
    'sentiment_analysis_server', 'get_predict_string')

pre = Predict()


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    start = time.time()
    print("进入predict函数")
    try:
        pre_string = str(flask.request.form.get(GET_PREDICT_STRING))
        print("输入参数为： {0}".format(pre_string))
        res = pre.predict(pre_string)
        return res
    except Exception as e:
        print(e)


@app.route('/test', methods=['GET'])
def test():
    print('connetcion success.')
    return 'success'


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=int(FLASK_PORT), threaded=True)
