#!/usr/bin/env python
# coding:utf-8

import httplib
import urllib
import urllib2
import configparser

config_path = "sentiment_analysis.config"
cf = configparser.ConfigParser()
cf.read(config_path)
FLASK_HOST = cf.get('sentiment_analysis_server', 'flask_host')
FLASK_PORT = cf.get('sentiment_analysis_server', 'flask_port')

# SENTIMENT_ANALYSIS_SERVER_URL = "http://" + FLASK_HOST + ":" + FLASK_PORT + "/predit"
# print(SENTIMENT_ANALYSIS_SERVER_URL)

def getPrediction(pre_string):
    SENTIMENT_ANALYSIS_SERVER_URL = "http://192.168.52.222:7330/predict"
    try:
        params = {'pre_string': pre_string}
        params_urlencode = urllib.urlencode(params)

        request = urllib2.Request(SENTIMENT_ANALYSIS_SERVER_URL, params_urlencode)
        response = urllib2.urlopen(request)
        res = response.read()
        if response.code != 200:
            return ('error code:' + response.code)
        else:
            print(res)
            return res
    except Exception as e:
        print(e)

    #
    # http_client.request('POST', params_urlencode)
    # r = http_client.getresponse()
    # print  r.status
    # print r.read()


if __name__ == '__main__':
    while True:
        pre_string = raw_input("please input: \n")
        if pre_string == "end":
            break
        res = getPrediction(pre_string)
