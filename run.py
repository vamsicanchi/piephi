# Python Imports 
import os
import sys
import json
import math
import random
import warnings
from typing import Optional
from datetime import datetime
from itertools import product
from collections import Counter
from pprint import pprint

# Library Imports

# Custom Imports
from appconfig import properties
from library.file.analyze import analyze_path
# Gloabal Variable/Settings

# try:
#     assert(os.getcwd()=="./")
# except AssertionError:
#     os.chdir("./")

if __name__ == "__main__":
    path_metadata = analyze_path("D:\code\python\piephi\samples")
    with open("temp.json", "w") as json_file:
        json_file.write(json.dumps(path_metadata))