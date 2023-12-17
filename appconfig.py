# Python Imports 
import os
import sys
import json
import math
import random
import warnings
from configparser import ConfigParser
from typing import Optional
from datetime import datetime
from itertools import product
from collections import Counter
from pprint import pprint

# Library Imports
import torch

# Custom Imports
from library.utils import read


# Gloabal Variable/Settings
settings = ConfigParser()
settings.read('settings.ini')
properties = read.read_json(os.path.join(settings.get('PATHS','base_path'), settings.get('PATHS','properties_path')))

anchors                             = [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                                       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                                       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]]
scales                              = [properties["dl"]["yolov3"]["image_size"] // 32, 
                                       properties["dl"]["yolov3"]["image_size"] // 16, 
                                       properties["dl"]["yolov3"]["image_size"] // 8]
device                              = "cuda" if torch.cuda.is_available() else "cpu"
properties["device"]                    = device
properties["dl"]["yolov3"]["anchors"]   = anchors
properties["dl"]["yolov3"]["scales"]    = scales