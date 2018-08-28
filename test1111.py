import numpy as np
import os
if not os.path.isfile('models/xx.txt'):
    print ("----")
    file = open('models/xx.txt', 'w')
    file.close()
# if not os.path.isfile(labels_file_name):
#     file = open(labels_file_name, 'w')
#     file.close()