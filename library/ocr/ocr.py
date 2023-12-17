# coding: utf-8

# Python Imports
import os
import re
import sys
import json
import glob
import time
import math
import timeit
import pathlib
import datetime
import dateutil
import tempfile
from sys import exc_info
from traceback import format_exception

# Library Imports
import PIL
import cv2 as cv
import numpy as np
import pandas as pd
import camelot
import pdf2image
import pytesseract
import pdfminer
import pdfplumber
import ocrmypdf
import camelot
import easyocr
import paddleocr



def print_exception():
    etype, value, tb = exc_info()
    info, error = format_exception(etype, value, tb)[-2:]
    print(f'Exception in:\n{info}\n{error}')

class Tesseract:

    """
    tesseract-ocr class for extracting text from images along with confidence, coordinates of each word in python
    """
    
    def __init__(self, tesseract_config):
        
        """
        Initialization to retrieve the user defined config to tesseract-ocr v4.1.1 & use as functional argument when calling pytesseract library

        config_string: ocr text extraction configuration parameters dictionary
                        {
                            "oem"                       : "",
                            "psm"                       : "",
                            "correct_osd"               : boolean,
                            "language"                  : "",
                            "landetect"                 : boolean,
                            "tessdata_dir"              : "",
                            "char_whitelist"            : "",
                            "char_blacklist"            : "",
                            "preserve_interword_spaces" : boolean 
                        } 
        config_data: ocr words confidence & coordinates configuration parameters dictionary
                        {
                            "oem"      : "",
                            "psm"      : "",
                            "language" : ""
                        } 
        """

        self.config_string  = tesseract_config["config_string"]
        self.config_data    = tesseract_config["config_data"]
        self.config_osd     = tesseract_config["config_osd"]
        
        self.string_config  = ""
        self.data_config    = ""

        try:
            
            if self.config_string["language"]!="":
                str_lang = " -l "+self.config_string["language"]
            else:
                str_lang = ""
        
            if self.config_string["oem"]!="":
                str_oem = " --oem "+self.config_string["oem"]
            else:
                str_oem = ""
            
            if self.config_string["psm"]!="":
                str_psm = " --psm "+self.config_string["psm"]
            else:
                str_psm = ""
            
            if self.config_string["char_whitelist"]!="":
                str_whitelist = " -c tessdata_char_whitelist "+self.config_string["char_whitelist"]
            else:
                str_whitelist = ""
            
            if self.config_string["char_blacklist"]!="":
                str_blacklist = " -c tessdata_char_blacklist "+self.config_string["char_blacklist"]
            else:
                str_blacklist = ""
            
            if self.config_string["tessdata_dir"]!="":
                str_tessdir = " --tessdata-dir "+self.config_string["tessdata_dir"]
            else:
                str_tessdir = ""

            if self.config_string["preserve_interword_spaces"]:
                str_preservespaces = " -c tessdata_char_blacklist=1 "
            else:
                str_preservespaces = ""

            self.string_config  =  str_lang + str_psm + str_oem  + str_preservespaces + str_whitelist + str_blacklist + str_tessdir

            if self.config_data["language"]!="":
                data_lang = " -l "+self.config_data["language"]
            else:
                data_lang = ""
        
            if self.config_data["oem"]!="":
                data_oem = " --oem "+self.config_data["oem"]
            else:
                data_oem = ""
            
            if self.config_data["psm"]!="":
                data_psm = " --psm "+self.config_data["psm"]
            else:
                str_psm = ""

            self.data_config = data_lang + data_psm + data_oem

        except Exception as excep:
            print_exception()

    def tesseract_osd(self, image):

        """
        Function to correct image orietation by calling pytesseract library

        config_osd: ocr words confidence & coordinates configuration parameters dictionary
                        {
                            "center" : "",
                            "scale"  : ""
                        } 
        """
        
        try:
            osd = pytesseract.image_to_osd(image)

            angle = 360 - int(re.search('(?<=rotate: )\d+', osd.lower()).group(0))
            script = re.search('(?<=script: )[a-z]+', osd.lower()).group(0)

            (h,w) = image.shape[:2]

            if self.config_osd["center"] is None:
                center = (w/2,h/2)
            elif isinstance(self.config_osd["center"], list):
                center = tuple(self.config_osd["center"])

            transformation_matrix = cv.getRotationMatrix2D(center, angle, self.config_osd["scale"])
            rotated_image = cv.warpAffine(image, transformation_matrix, (w, h))
            image = rotated_image

        except Exception as excep:
            etype, value, tb = exc_info()
            info, error = format_exception(etype, value, tb)[-2:]
            print(f'Exception in:\n{info}\n{error}')

        return image

    def tesseract_ocr(self, image):
        """
        Function to run tesseract-ocr on given image and extract whole text in string format with user defined tesseract config

        string_config: from class instantiation

        returns python string

        """
        
        self.extracted_string = ""

        try:
            self.extracted_string = pytesseract.image_to_string(image, config=self.string_config)
        except Exception as excep:
            print_exception()

        return self.extracted_string

    def tesseract_ocr_metadata(self, image):
        """
        Function to run tesseract-ocr on given image and get confidence & coordinates of each word in image with user defined tesseract config

        data_config: from class instantiation

        returns python dictionary

        """
        
        self.extracted_data = {}

        try:
            self.extracted_data = pytesseract.image_to_data(image, config=self.data_config, output_type=pytesseract.Output.DICT)
        except Exception as excep:
            print_exception()

        return self.extracted_data

    def tesseract_searchable_pdf(self, image):
        """
        Function to convert an image to searchable pdf
        """
        
        try:
            searchable_pdf = pytesseract.image_to_pdf_or_hocr(image, config=self.string_config, extension='pdf')
        except Exception as excep:
            print("Exception in searchable pdf function")
            print_exception()

        return searchable_pdf

    def tesseract_string_to_dict(self, extracted_string):
        
        """
        Function to covert ocr'd text from one image into python dictionary with line number as keys

        extracted_string: python string with all text from each image line by line

        returns python dictionary

        """

        self.result_dict = {}
        line_number = 0
        try:
            data = self.extracted_string.split("\n")
            for line in data:
                if line==" " or line=="":
                    pass
                else:
                    line_number+=1
                    self.result_dict[line_number]=line
        except Exception as  excep:
            print_exception()

        return self.result_dict

    def run_tesseract(self, image, image_path):

        image_osd      = self.tesseract_osd(image)
        text_data      = self.tesseract_ocr(image_osd)
        text_data_dict = self.tesseract_string_to_dict(text_data)
        text_metadata  = self.tesseract_ocr_metadata(image_osd)
        searchable_pdf = self.tesseract_searchable_pdf(image_path) 

        return image_osd, text_data, text_data_dict, text_metadata, searchable_pdf

class Paddle:
    
    def __init__(self, paddle_config, log) -> None:
        self._paddle_config = paddle_config
        self._log = log
        self._ocr = paddleocr.PaddleOCR(
                                            alpha                   = paddle_config["alpha"],
                                            benchmark               = paddle_config["benchmark"],
                                            beta                    = paddle_config["beta"],
                                            cls_batch_num           = paddle_config["cls_batch_num"],
                                            cls_image_shape         = paddle_config["cls_image_shape"],
                                            cls_model_dir           = paddle_config["cls_model_dir"],
                                            cls_thresh              = paddle_config["cls_thresh"],
                                            cpu_threads             = paddle_config["cpu_threads"],
                                            crop_res_save_dir       = paddle_config["crop_res_save_dir"],
                                            det                     = paddle_config["det"],
                                            det_algorithm           = paddle_config["det_algorithm"],
                                            det_box_type            = paddle_config["det_box_type"],
                                            det_db_box_thresh       = paddle_config["det_db_box_thresh"],
                                            det_db_score_mode       = paddle_config["det_db_score_mode"],
                                            det_db_thresh           = paddle_config["det_db_thresh"],
                                            det_db_unclip_ratio     = paddle_config["det_db_unclip_ratio"],
                                            det_east_cover_thresh   = paddle_config["det_east_cover_thresh"],
                                            det_east_nms_thresh     = paddle_config["det_east_nms_thresh"],
                                            det_east_score_thresh   = paddle_config["det_east_score_thresh"],
                                            det_limit_side_len      = paddle_config["det_limit_side_len"],
                                            det_limit_type          = paddle_config["det_limit_type"],
                                            det_model_dir           = paddle_config["det_model_dir"],
                                            det_pse_box_thresh      = paddle_config["det_pse_box_thresh"],
                                            det_pse_min_area        = paddle_config["det_pse_min_area"],
                                            det_pse_scale           = paddle_config["det_pse_scale"],
                                            det_pse_thresh          = paddle_config["det_pse_thresh"],
                                            det_sast_nms_thresh     = paddle_config["det_sast_nms_thresh"],
                                            det_sast_score_thresh   = paddle_config["det_sast_score_thresh"],
                                            draw_img_save_dir       = paddle_config["draw_img_save_dir"],
                                            drop_score              = paddle_config["drop_score"],
                                            e2e_algorithm           = paddle_config["e2e_algorithm" ],
                                            e2e_char_dict_path      = paddle_config["e2e_char_dict_path"],
                                            e2e_limit_side_len      = paddle_config["e2e_limit_side_len"],
                                            e2e_limit_type          = paddle_config["e2e_limit_type"],
                                            e2e_model_dir           = paddle_config["e2e_model_dir"],
                                            e2e_pgnet_mode          = paddle_config["e2e_pgnet_mode"],
                                            e2e_pgnet_score_thresh  = paddle_config["e2e_pgnet_score_thresh"],
                                            e2e_pgnet_valid_set     = paddle_config["e2e_pgnet_valid_set"],
                                            enable_mkldnn           = paddle_config["enable_mkldnn"],
                                            fourier_degree          = paddle_config["fourier_degree"],
                                            gpu_mem                 = paddle_config["gpu_mem"],
                                            help                    = paddle_config["help"],
                                            image_dir               = paddle_config["image_dir"],
                                            image_orientation       = paddle_config["image_orientation"],
                                            ir_optim                = paddle_config["ir_optim"],
                                            kie_algorithm           = paddle_config["kie_algorithm"],
                                            label_list              = paddle_config["label_list"],
                                            lang                    = paddle_config["lang"],
                                            layout                  = paddle_config["layout"],
                                            layout_dict_path        = paddle_config["layout_dict_path"],
                                            layout_model_dir        = paddle_config["layout_model_dir"],
                                            layout_nms_threshold    = paddle_config["layout_nms_threshold"],
                                            layout_score_threshold  = paddle_config["layout_score_threshold"],
                                            max_batch_size          = paddle_config["max_batch_size"],
                                            max_text_length         = paddle_config["max_text_length"],
                                            merge_no_span_structure = paddle_config["merge_no_span_structure"],
                                            min_subgraph_size       = paddle_config["min_subgraph_size"],
                                            mode                    = paddle_config["mode"],
                                            ocr                     = paddle_config["ocr"],
                                            ocr_order_method        = paddle_config["ocr_order_method"],
                                            ocr_version             = paddle_config["ocr_version"],
                                            output                  = paddle_config["output"],
                                            page_num                = paddle_config["page_num"],
                                            precision               = paddle_config["precision"],
                                            process_id              = paddle_config["process_id"],
                                            re_model_dir            = paddle_config["re_model_dir"],
                                            rec                     = paddle_config["rec"],
                                            rec_algorithm           = paddle_config["rec_algorithm"],
                                            rec_batch_num           = paddle_config["rec_batch_num"],
                                            rec_char_dict_path      = paddle_config["rec_char_dict_path"],
                                            rec_image_inverse       = paddle_config["rec_image_inverse"],
                                            rec_image_shape         = paddle_config["rec_image_shape"],
                                            rec_model_dir           = paddle_config["rec_model_dir"],
                                            recovery                = paddle_config["recovery"],
                                            save_crop_res           = paddle_config["save_crop_res"],
                                            save_log_path           = paddle_config["save_log_path"],
                                            scales                  = paddle_config["scales"],
                                            ser_dict_path           = paddle_config["ser_dict_path"],
                                            ser_model_dir           = paddle_config["ser_model_dir"],
                                            show_log                = paddle_config["show_log"],
                                            sr_batch_num            = paddle_config["sr_batch_num"],
                                            sr_image_shape          = paddle_config["sr_image_shape"],
                                            sr_model_dir            = paddle_config["sr_model_dir"],
                                            structure_version       = paddle_config["structure_version"],
                                            table                   = paddle_config["table"],
                                            table_algorithm         = paddle_config["table_algorithm"],
                                            table_char_dict_path    = paddle_config["table_char_dict_path"],
                                            table_max_len           = paddle_config["table_max_len"],
                                            table_model_dir         = paddle_config["table_model_dir"],
                                            total_process_num       = paddle_config["total_process_num"],
                                            type                    = paddle_config["type"],
                                            use_angle_cls           = paddle_config["use_angle_cls"],
                                            use_dilation            = paddle_config["use_dilation"],
                                            use_gpu                 = paddle_config["use_gpu"],
                                            use_mp                  = paddle_config["use_mp"],
                                            use_npu                 = paddle_config["use_npu"],
                                            use_onnx                = paddle_config["use_onnx"],
                                            use_pdf2docx_api        = paddle_config["use_pdf2docx_api"],
                                            use_pdserving           = paddle_config["use_pdserving"],
                                            use_space_char          = paddle_config["use_space_char"],
                                            use_tensorrt            = paddle_config["use_tensorrt"],
                                            use_visual_backbone     = paddle_config["use_visual_backbone"],
                                            use_xpu                 = paddle_config["use_xpu"],
                                            vis_font_path           = paddle_config["vis_font_path"],
                                            warmup                  = paddle_config["warmup"],
                                        )
    
    def run_paddle(self, image):
        result = self._ocr.ocr(image)
        return result

class Easyocr:
    
    def __init__(self, easyocr_config, log) -> None:
        
        self._reader    = easyocr.Reader(
            lang_list               = easyocr_config["lang_list"],
            gpu                     = easyocr_config["gpu"],
            model_storage_directory = easyocr_config["model_storage_directory"],
            download_enabled        = easyocr_config["download_enabled"],
            user_network_directory  = easyocr_config["user_network_directory"],
            detector                = easyocr_config["detector"],
            recognizer              = easyocr_config["recognizer"],

        )
    
    def run_easyocr(self, image):
        result = self._reader

class Extract:
    
    def __init__(self, path_config, extraction_config):
        
        self._searchable_pdf_temp = path_config["searchable_pdf_temp"]
        self._text_file_temp      = path_config["text_file_temp"]
        self._ocr_my_pdf          = extraction_config["ocrmypdf"]
        self._camelot             = extraction_config["camelot"]

    def searchable_pdf_temp(self, searchable_pdf):

        try:
            tempfile.tempdir = self._searchable_pdf_temp
            _, temp_file = tempfile.mkstemp(suffix = '.pdf')
            with open(temp_file,'w+b') as fw:
                fw.write(searchable_pdf)

        except Exception as excep:
            print("\n exception in searchable_pdf_temp function \n")
            print_exception()

        return temp_file

    def apply_ocrmypdf(self, image_path):

        try:
            tempfile.tempdir = self._searchable_pdf_temp
            _, temp_file     = tempfile.mkstemp(suffix = '.pdf')
            ocrmypdf.ocr(image_path, 
                         temp_file, 
                         language               = self._ocr_my_pdf["language"],
                         output_type            = self._ocr_my_pdf["output_type"],
                         sidecar                = self._ocr_my_pdf["sidecar"],
                         jobs                   = self._ocr_my_pdf["jobs"],
                         use_threads            = self._ocr_my_pdf["use_threads"],
                         title                  = self._ocr_my_pdf["title"],
                         author                 = self._ocr_my_pdf["author"],
                         subject                = self._ocr_my_pdf["subject"],
                         keywords               = self._ocr_my_pdf["keywords"],
                         rotate_pages           = self._ocr_my_pdf["rotate_pages"],
                         image_dpi              = self._ocr_my_pdf["image_dpi"],
                         remove_background      = self._ocr_my_pdf["remove_background"],
                         deskew                 = self._ocr_my_pdf["deskew"],
                         clean                  = self._ocr_my_pdf["clean"],
                         clean_final            = self._ocr_my_pdf["clean_final"],
                         unpaper_args           = self._ocr_my_pdf["unpaper_args"],
                         oversample             = self._ocr_my_pdf["oversample"],
                         remove_vectors         = self._ocr_my_pdf["remove_vectors"],
                         threshold              = self._ocr_my_pdf["threshold"],
                         force_ocr              = self._ocr_my_pdf["force_ocr"],
                         skip_text              = self._ocr_my_pdf["skip_text"],
                         redo_ocr               = self._ocr_my_pdf["redo_ocr"],
                         skip_big               = self._ocr_my_pdf["skip_big"],
                         optimize               = self._ocr_my_pdf["optimize"],
                         jpg_quality            = self._ocr_my_pdf["jpg_quality"],
                         png_quality            = self._ocr_my_pdf["png_quality"],
                         jbig2_lossy            = self._ocr_my_pdf["jbig2_lossy"],
                         jbig2_page_group_size  = self._ocr_my_pdf["jbig2_page_group_size"],
                         pages                  = self._ocr_my_pdf["pages"],
                         max_image_mpixels      = self._ocr_my_pdf["max_image_mpixels"],
                         tesseract_config       = self._ocr_my_pdf["tesseract_config"],
                         tesseract_pagesegmode  = self._ocr_my_pdf["tesseract_pagesegmode"],
                         tesseract_oem          = self._ocr_my_pdf["tesseract_oem"],
                         pdf_renderer           = self._ocr_my_pdf["pdf_renderer"],
                         tesseract_timeout      = self._ocr_my_pdf["tesseract_timeout"],
                         rotate_pages_threshold = self._ocr_my_pdf["rotate_pages_threshold"],
                         pdfa_image_compression = self._ocr_my_pdf["pdfa_image_compression"],
                         user_words             = self._ocr_my_pdf["user_words"],
                         user_patterns          = self._ocr_my_pdf["user_patterns"],
                         fast_web_view          = self._ocr_my_pdf["fast_web_view"],
                         plugins                = self._ocr_my_pdf["plugins"],
                         plugin_manager         = self._ocr_my_pdf["plugin_manager"],
                         keep_temporary_files   = self._ocr_my_pdf["keep_temporary_files"],
                         progress_bar           = self._ocr_my_pdf["progress_bar"]                      
                        )

        except Exception as excep:
            print("\n exception in apply_ocrmypdf function \n")
            print_exception()

        return temp_file
        
    def apply_camelot(self, temp_pdf_path):
        
        tables = camelot.read_pdf(temp_pdf_path, flavor='stream') 

        return tables

    def apply_pdfplumber(self, file_path):

        pdf = pdfplumber.open()
