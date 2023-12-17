# Python Imports
import os
import sys
import time
import copy
import json
import shutil
from pprint import pprint

# Library Imports
import rasterio

# Gloabal Variable/Settings
sys.path.append("D:\\igis\\aiml\\core")

# Custom Imports
from source.appconfig import config
from library.utils.log import log
from library.yolo.yolov3 import yolov3_inference, yolov3_inference_api, yolov3_inference_statistics
from library.gis.geoconversions import convert_tif_to_byte_gdal, split_image_rasterio, RasterioCRSTransformer
from library.image.opencv import resize_image_opencv_pad

geojson_template = {
    "features": [],
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {
            "name": "epsg:4326"
        }
    }
}

feature_template = {
    "type": "Feature",
    "geometry": {
        "coordinates": [[[],[],[],[],[]]],
        "type": "Polygon"
    },
    "properties": {
        "confidence_score": 0.0,
        "class_name": "",
    }
}

def images_to_tiles(config, dataset_name, root, file, dir_path, tiles_path):

    convert_tif_to_byte_gdal(os.path.join(root, file), dir_path, ".png")
                
    png_filename  = file.split(".")[0]+".png"
    
    resize_image_opencv_pad(os.path.join(root, png_filename), 
                            os.path.join(root, png_filename), 
                            resize_image_width=6240, 
                            resize_image_height=6240, 
                            color=(255,255,255))
    
    split_image_rasterio(os.path.join(root, png_filename), 
                         os.path.join(dir_path, tiles_path), 
                         config["datasets"][dataset_name]["tile_name_format"], 
                         width_offset=(False,int(config["dl"]["yolov3"]["image_size"]/2)), 
                         height_offset=(False,int(config["dl"]["yolov3"]["image_size"]/2)), 
                         tile_width=config["dl"]["yolov3"]["image_size"], 
                         tile_height=config["dl"]["yolov3"]["image_size"], 
                         fill_value=255)

def run_inference(config, dataset_name, log):
    
    def inference(config, dataset_name, dir_path, tiles_path, root, file, log):
        inference_start_time = time.time()
        inference_image = os.path.join(root, file)
        log.info("Running inference on..."+inference_image, task="inference")
        log.info("Converting tif image to png, resize to standard size, converting to tiles...", task="inference")
        images_to_tiles(config, dataset_name, root, file, dir_path, tiles_path)
        inference_image_png, inference_image_bbox, inference_bboxes = yolov3_inference(config, dataset_name, inference_image, log)
        log.info("Removing temp tiles folder...", task="inference")             
        shutil.rmtree(os.path.join(dir_path, tiles_path))
        inference_elapsed_time = time.time() - inference_start_time
        log.info("Inference...execution time on {image}: ".format(image=inference_image)+ str(time.strftime("%H:%M:%S", time.gmtime(inference_elapsed_time)))+" HH:MM:SS ", task="inference")

        return inference_image_png, inference_image_bbox, inference_bboxes

    def create_geojson(config, inference_image_bbox, inference_bboxes, root, file, log):

        geojson = copy.deepcopy(geojson_template)

        with rasterio.open(os.path.join(root, file)) as source:
            for bbox in inference_bboxes:
                feature = copy.deepcopy(feature_template)
                for key, value in bbox.items():
                    pixelcoord_to_geocoord = RasterioCRSTransformer(source.transform, source.crs.wkt)
                    if key in ["bbox_ul", "bbox_ur", "bbox_br", "bbox_bl"]:    
                        geocoord = pixelcoord_to_geocoord.pixel_to_map(value)
                        if key=="bbox_ul":
                            feature["geometry"]["coordinates"][0][0] = list(geocoord)
                            feature["geometry"]["coordinates"][0][4] = list(geocoord)
                        if key=="bbox_ur":
                            feature["geometry"]["coordinates"][0][1] = list(geocoord)
                        if key=="bbox_br":
                            feature["geometry"]["coordinates"][0][2] = list(geocoord)
                        if key=="bbox_bl":
                            feature["geometry"]["coordinates"][0][3] = list(geocoord)
                    if key=="confidence":
                        feature["properties"]["confidence_score"] = value
                    if key=="class_name":
                        feature["properties"]["class_name"] = value
                geojson["features"].append(feature)
        
        with open(inference_image_bbox.replace(".png", ".geojson"), 'w') as pred_bboxes:
            pred_bboxes.write(json.dumps(geojson))
    
    dir_path   = config["datasets"][dataset_name]["inference_dir"]  
    tiles_path = config["datasets"][dataset_name]["tile_dir"]
    
    for root, dirs, files in os.walk(dir_path):       
        for file in files:  
            if os.path.join(root, file).endswith((".tif")):
                inference_image_png, inference_image_bbox, inference_bboxes = inference(config, dataset_name, dir_path, tiles_path, root, file, log)
                create_geojson(config, inference_image_bbox, inference_bboxes, root, file, log)

                break

def run_inference_statistics(config, dataset_name, log):
    
    def inference(config, model_filename, dataset_name, dir_path, tiles_path, root, file, log):
        inference_start_time = time.time()
        inference_image = os.path.join(root, file)
        log.info("Running inference on..."+inference_image, task="inference")
        log.info("Converting tif image to png, resize to standard size, converting to tiles...", task="inference")
        images_to_tiles(config, dataset_name, root, file, dir_path, tiles_path)
        inference_image_png, inference_image_bbox, inference_bboxes = yolov3_inference_statistics(config, model_filename, dataset_name, inference_image, log)
        log.info("Removing temp tiles folder...", task="inference")             
        shutil.rmtree(os.path.join(dir_path, tiles_path))
        inference_elapsed_time = time.time() - inference_start_time
        log.info("Inference...execution time on {image}: ".format(image=inference_image)+ str(time.strftime("%H:%M:%S", time.gmtime(inference_elapsed_time)))+" HH:MM:SS ", task="inference")

        return inference_image_png, inference_image_bbox, inference_bboxes

    def create_geojson(config, inference_image_bbox, inference_bboxes, root, file, log):

        geojson = copy.deepcopy(geojson_template)

        with rasterio.open(os.path.join(root, file)) as source:
            for bbox in inference_bboxes:
                feature = copy.deepcopy(feature_template)
                for key, value in bbox.items():
                    pixelcoord_to_geocoord = RasterioCRSTransformer(source.transform, source.crs.wkt)
                    if key in ["bbox_ul", "bbox_ur", "bbox_br", "bbox_bl"]:    
                        geocoord = pixelcoord_to_geocoord.pixel_to_map(value)
                        if key=="bbox_ul":
                            feature["geometry"]["coordinates"][0][0] = list(geocoord)
                            feature["geometry"]["coordinates"][0][4] = list(geocoord)
                        if key=="bbox_ur":
                            feature["geometry"]["coordinates"][0][1] = list(geocoord)
                        if key=="bbox_br":
                            feature["geometry"]["coordinates"][0][2] = list(geocoord)
                        if key=="bbox_bl":
                            feature["geometry"]["coordinates"][0][3] = list(geocoord)
                    if key=="confidence":
                        feature["properties"]["confidence_score"] = value
                    if key=="class_name":
                        feature["properties"]["class_name"] = value
                geojson["features"].append(feature)
        
        with open(inference_image_bbox.replace(".png", ".geojson"), 'w') as pred_bboxes:
            pred_bboxes.write(json.dumps(geojson))
    
    model_path      = os.path.dirname(config["datasets"][dataset_name]["save_checkpoint_file"])
    dir_path        = config["datasets"][dataset_name]["inference_dir"]  
    tiles_path      = config["datasets"][dataset_name]["tile_dir"]
    
    for model_root, model_dirs, model_files in os.walk(model_path):
        for model_file in model_files:
            if "epoch" in os.path.join(model_root, model_file) and model_file not in ["toppotsdam-v5-epoch-10-mAP-57.pth", "toppotsdam-v5-epoch-102-mAP-56.pth", "toppotsdam-v5-epoch-105-mAP-63.pth", "toppotsdam-v5-epoch-115-mAP-55.pth"]:
                log.info("Using model..."+model_file+"..................................................", task="inference")
                config["datasets"][dataset_name]["save_checkpoint_file"] = os.path.join(model_root, model_file)
                model_filename  = model_file.replace(".pth","")    
                for root, dirs, files in os.walk(dir_path):       
                    for file in files:  
                        if os.path.join(root, file).endswith((".tif")):
                            inference_image_png, inference_image_bbox, inference_bboxes = inference(config, model_filename, dataset_name, dir_path, tiles_path, root, file, log)
                            create_geojson(config, inference_image_bbox, inference_bboxes, root, file, log)

def image_to_tiles(config, dataset_name, image_file, image_dir, tiles_path):

    convert_tif_to_byte_gdal(image_file, image_dir, ".png")
                
    png_filename  = image_file.split(".")[0]+".png"
    
    resize_image_opencv_pad(png_filename, 
                            png_filename, 
                            resize_image_width=6240, 
                            resize_image_height=6240, 
                            color=(255,255,255))
    
    split_image_rasterio(png_filename, 
                         os.path.join(image_dir, tiles_path), 
                         config["datasets"][dataset_name]["tile_name_format"], 
                         width_offset=(False,int(config["dl"]["yolov3"]["image_size"]/2)), 
                         height_offset=(False,int(config["dl"]["yolov3"]["image_size"]/2)), 
                         tile_width=config["dl"]["yolov3"]["image_size"], 
                         tile_height=config["dl"]["yolov3"]["image_size"], 
                         fill_value=255)

def run_inference_image(config, dataset_name, image_file, image_dir, log):
    
    def inference(config, dataset_name, inference_image, image_dir, tiles_path, log):
        
        inference_start_time = time.time()
        log.info("Running inference on..."+inference_image, task="inference")
        log.info("Converting tif image to png, resize to standard size, converting to tiles...", task="inference")
        image_to_tiles(config, dataset_name, inference_image, image_dir, tiles_path)
        inference_image_png, inference_image_bbox, inference_bboxes = yolov3_inference_api(config, dataset_name, inference_image, image_dir, log)
        log.info("Removing temp tiles folder...", task="inference")             
        shutil.rmtree(os.path.join(image_dir, tiles_path))
        inference_elapsed_time = time.time() - inference_start_time
        log.info("Inference...execution time on {image}: ".format(image=inference_image)+ str(time.strftime("%H:%M:%S", time.gmtime(inference_elapsed_time)))+" HH:MM:SS ", task="inference")

        return inference_image_png, inference_image_bbox, inference_bboxes

    def create_geojson(config, inference_image_bbox, inference_bboxes, image_file, log):

        geojson = copy.deepcopy(geojson_template)

        with rasterio.open(image_file) as source:
            for bbox in inference_bboxes:
                feature = copy.deepcopy(feature_template)
                for key, value in bbox.items():
                    pixelcoord_to_geocoord = RasterioCRSTransformer(source.transform, source.crs.wkt)
                    if key in ["bbox_ul", "bbox_ur", "bbox_br", "bbox_bl"]:    
                        geocoord = pixelcoord_to_geocoord.pixel_to_map(value)
                        if key=="bbox_ul":
                            feature["geometry"]["coordinates"][0][0] = list(geocoord)
                            feature["geometry"]["coordinates"][0][4] = list(geocoord)
                        if key=="bbox_ur":
                            feature["geometry"]["coordinates"][0][1] = list(geocoord)
                        if key=="bbox_br":
                            feature["geometry"]["coordinates"][0][2] = list(geocoord)
                        if key=="bbox_bl":
                            feature["geometry"]["coordinates"][0][3] = list(geocoord)
                    if key=="confidence":
                        feature["properties"]["confidence_score"] = value
                    if key=="class_name":
                        feature["properties"]["class_name"] = value
                geojson["features"].append(feature)
        
        with open(inference_image_bbox.replace(".png", ".geojson"), 'w') as pred_bboxes:
            pred_bboxes.write(json.dumps(geojson))
  
    tiles_path = config["datasets"][dataset_name]["tile_dir"]
    inference_image_png, inference_image_bbox, inference_bboxes = inference(config, dataset_name, image_file, image_dir, tiles_path, log)
    create_geojson(config, inference_image_bbox, inference_bboxes, image_file, log)

if __name__=='__main__':   
    # run_inference(config, "toppotsdam", log)
    # run_inference_statistics(config, "toppotsdam", log)
    run_inference_image(config, "toppotsdam", r"D:\datasets\sgl\toppotsdam\inferenceapi\source\top_potsdam_3_12_RGBIR.tif", r"D:\datasets\sgl\toppotsdam\inferenceapi\source", log)