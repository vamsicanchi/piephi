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
from library.yolo.yolov3 import yolov3_test

def run_test(config, dataset_name, log):
    
    dir_path   = config["datasets"][dataset_name]["test_dir"]  
    tiles_path = config["datasets"][dataset_name]["tile_dir"]
    
    for root, dirs, files in os.walk(dir_path):
        
        for file in files:
          
            if os.path.join(root, file).endswith((".tif")):
                
                test_start_time = time.time()
                test_image = os.path.join(root, file)
                log.info("Running test on..."+test_image, task="test")
                yolov3_test(config, dataset_name, test_image, log)
                log.info("Removing temp tiles folder...", task="test")             
                shutil.rmtree(os.path.join(dir_path, tiles_path))
                test_elapsed_time = time.time() - test_start_time
                log.info("Inference...execution time on {image}: ".format(image=test_image)+ str(time.strftime("%H:%M:%S", time.gmtime(test_elapsed_time)))+" HH:MM:SS ", task="test")
                
if __name__=='__main__':
    run_test(config, "toppotsdam", log)