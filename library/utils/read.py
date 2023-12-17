# Python Imports 
import os
import sys
import json
import orjson
import simplejson

# Library Imports

# Custom Imports
from library.utils.log import applog

# Gloabal Variable/Settings



# Function to read json from given path as python dictionary
def read_json(json_path: str) -> dict:
    """
    Function to read json from given path as python dictionary
    Args:
        json_path (str): Provide full json path

    Returns:
        dict: returns python dictionary
    """

    try:
        with open(json_path,'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        applog.catcherror( message="Properties JSON file not found. Check the path in settings.ini - "+json_path, task="exception")
        exit()      

    return json_data