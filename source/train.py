# Python Imports
import os
import sys
import time
import copy
import json
import shutil

# Library Imports

# Gloabal Variable/Settings
sys.path.append("D:\\igis\\aiml\\core")

# Custom Imports
from appconfig import config
from library.utils.log import log
from library.yolo.yolov3 import yolov3_train

if __name__=='__main__':
    yolov3_train(config, "toppotsdam", log)