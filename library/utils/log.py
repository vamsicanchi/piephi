# Python Imports
import os
import sys
from configparser import ConfigParser

# Library Imports
from loguru import logger

# Custom Imports

# Gloabal Variable/Settings
sys.tracebacklimit = 2

class AppLogger():
    
    def __init__(self, base_path, log_path):
        
        self._format    = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {process} | {message}"
        self._colorize  = False
        self._encoding  = "utf-8"
        self._enqueue   = True
        
        self.api = logger.bind(task="api")
        logger.add( 
                    os.path.join(base_path, log_path, "api.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue, 
                    filter=lambda record: record["extra"]["task"] == "api"
                  )
        
        self.process = logger.bind(task="process")
        logger.add( 
                    os.path.join(base_path, log_path, "process.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue,
                    filter=lambda record: record["extra"]["task"] == "process"
                  )
        
        self.catch = logger.bind(task="exception")
        logger.add( 
                    os.path.join(base_path, log_path, "exception.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue,
                    filter=lambda record: record["extra"]["task"] == "exception"
                  )

        self.ocrapi = logger.bind(task="ocrapi")
        logger.add( 
                    os.path.join(base_path, log_path, "ocrapi.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue, 
                    filter=lambda record: record["extra"]["task"] == "ocrapi"
                  )
        
        self.ocrprocess = logger.bind(task="ocrprocess")
        logger.add( 
                    os.path.join(base_path, log_path, "ocrprocess.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue,
                    filter=lambda record: record["extra"]["task"] == "ocrprocess"
                  )
        
        self.ocrcatch = logger.bind(task="ocrexception")
        logger.add( 
                    os.path.join(base_path, log_path, "ocrexception.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue,
                    filter=lambda record: record["extra"]["task"] == "ocrexception"
                  )

        self.train = logger.bind(task="train")
        logger.add(
                    os.path.join(base_path, log_path, "train.log"), 
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue, 
                    filter=lambda record: record["extra"]["task"] == "train"
                  )
        
        self.validation = logger.bind(task="validation")
        logger.add(
                    os.path.join(base_path, log_path, "validation.log"), 
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue, 
                    filter=lambda record: record["extra"]["task"] == "validation"
                  )
        
        self.test = logger.bind(task="test")
        logger.add(
                    os.path.join(base_path, log_path, "test.log"), 
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue, 
                    filter=lambda record: record["extra"]["task"] == "test"
                  )
        
        self.inference = logger.bind(task="inference")
        logger.add(
                    os.path.join(base_path, log_path, "inference.log"), 
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue, 
                    filter=lambda record: record["extra"]["task"] == "inference"
                  )
        
        self.modelcatch = logger.bind(task="modelexception")
        logger.add( 
                    os.path.join(base_path, log_path, "modelexception.log"),
                    format=self._format, 
                    colorize=self._colorize,
                    encoding=self._encoding, 
                    enqueue=self._enqueue,
                    filter=lambda record: record["extra"]["task"] == "modelexception"
                  )

    def info(self, message, task):
        
        if task=="api":
            self.api.info(message)
        
        if task=="process":
            self.process. info(message)
        
        if task=="ocrapi":
            self.ocrapi.info(message)
        
        if task=="ocrprocess":
            self.ocrprocess. info(message)
        
        if task=="train":
            self.train.info(message)
        
        if task=="validation":
            self.validation.info(message)
        
        if task=="test":
            self.test.info(message)
        
        if task=="inference":
            self.inference.info(message)

    def catcherror (self, message, task):
        if task=="exception":
            self.catch.exception(message, diagnose=False)
        if task=="ocrexception":
            self.ocrcatch.exception(message, backtrace=True, diagnose=True)
        if task=="modelexception":
            self.modelcatch.exception(message, backtrace=True, diagnose=True)

settings = ConfigParser()
settings.read('settings.ini')

applog   = AppLogger(settings.get('PATHS','base_path'), settings.get('PATHS','logs_path'))