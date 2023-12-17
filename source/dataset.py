# Python Imports
import os
import sys
import time
import copy
import json
from pprint import pprint

# Library Imports
import os
import cv2
import pandas as pd

# Gloabal Variable/Settings
# sys.path.append("D:\\igis\\aiml\\core")

# Custom Imports
from appconfig import config
from library.utils.log import log
from library.gis.geoconversions import convert_tif_to_byte_gdal, convert_geocoord_pixelcoord
from library.image.opencv import resize_image_opencv_pad

def get_tile_from_bbox(bbox_dict, image_width, image_height, tile_width, tile_height):

    tile_dict = {}

    tile_width_radius  = int(tile_width/2)
    tile_height_radius = int(tile_height/2)

    bbox_width  = bbox_dict["bbox_ur"][0] - bbox_dict["bbox_ul"][0]
    bbox_height = bbox_dict["bbox_bl"][1] - bbox_dict["bbox_ul"][1]

    bbox_x_center = int((bbox_dict["bbox_ul"][0] + bbox_dict["bbox_br"][0])/2)
    bbox_y_center = int((bbox_dict["bbox_ul"][1] + bbox_dict["bbox_br"][1])/2)


    ul_ur_midpoint = (bbox_x_center if bbox_x_center> tile_height_radius else 0, bbox_y_center-tile_height_radius if bbox_y_center>tile_height_radius else 0)
    br_bl_midpoint = (bbox_x_center if bbox_x_center> tile_width_radius else 0, bbox_y_center+tile_height_radius if bbox_y_center<image_height+tile_height_radius else 0)
    
    ur_br_midpoint = (bbox_x_center+tile_width_radius if bbox_x_center> tile_width_radius else 0, bbox_y_center if bbox_y_center<tile_height_radius else 0)
    br_ul_midpoint = (bbox_x_center-tile_width_radius if bbox_x_center> tile_width_radius else 0, bbox_y_center if bbox_y_center<tile_height_radius else 0)

    tile_ul = (ul_ur_midpoint[0]-tile_width_radius if ul_ur_midpoint[0]>tile_width_radius else 0, ul_ur_midpoint[1])
    tile_ur = (ul_ur_midpoint[0]+tile_width_radius, ul_ur_midpoint[1])
    tile_bl = (br_bl_midpoint[0]-tile_width_radius if br_bl_midpoint[0]>tile_width_radius else 0, br_bl_midpoint[1])
    tile_br = (br_bl_midpoint[0]+tile_width_radius, br_bl_midpoint[1])


    if tile_br[0]-tile_bl[0]!=tile_width or tile_ur[0]-tile_ul[0]!=tile_width:

        tile_ur = (tile_ul[0]+tile_width, tile_ur[1])
        tile_br = (tile_bl[0]+tile_width, tile_br[1])

    if tile_ur[1]==0 and tile_ul[1]==0:

        tile_bl = (tile_bl[0], tile_ul[1]+tile_height)
        tile_br = (tile_br[0], tile_ur[1]+tile_height)       

    tile_dict["tile_ul"] = tile_ul
    tile_dict["tile_ur"] = tile_ur
    tile_dict["tile_br"] = tile_br
    tile_dict["tile_bl"] = tile_bl

    return tile_dict

def split_image_to_tiles(input_image_path, bboxes, image_width, image_height, tile_width, tile_height, save_tiles, tiles_path):

    tiles = {}

    input_filename = os.path.basename(input_image_path).split(".")[0]

    for idx, bbox_dict in bboxes.items():
        
        tile_dict = get_tile_from_bbox(bbox_dict, image_width, image_height, tile_width, tile_height)
        tiles.update({str(idx):tile_dict})

    final_result    = {idx:[] for idx, val in tiles.items()}
    bbox_flag       = {idx:False for idx, val in tiles.items()}

    for tile_idx, tile_bbox in tiles.items():
        result = [] 
        final_result[str(tile_idx)].append(tile_bbox)
        
        for bbox_idx, bbox in bboxes.items():
            # If top-left inner box corner is inside the tile bounding box
            if  bbox['bbox_ul'][0]>tile_bbox['tile_ul'][0] and bbox['bbox_ul'][1]>tile_bbox['tile_ul'][1]:
                # If bottom-right inner box corner is inside the tile bounding box
                if bbox['bbox_br'][0]<tile_bbox['tile_br'][0] and bbox['bbox_br'][1]<tile_bbox['tile_br'][1]:
                    if not bbox_flag[bbox_idx]:
                        result.append(bbox)
                        bbox_flag[bbox_idx] = True

        final_result[str(tile_idx)].append(result)
    
    if save_tiles:
        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)

        image = cv2.imread(input_image_path)
        for idx, val in final_result.items():

            if len(val[1])>0:
                
                # image = cv2.rectangle(image, val[0]['tile_ul'], val[0]['tile_br'], config.TILE_COLOR, config.TILE_THICKNESS)
                # for i in range(len(val[1])):         
                    # image = cv2.rectangle(image, val[1][i]['bbox_ul'], val[1][i]['bbox_br'], config.BBOX_COLOR, config.BBOX_THICKNESS)

                tile_image = image[val[0]['tile_ul'][1]:val[0]['tile_br'][1], val[0]['tile_ul'][0]:val[0]['tile_br'][0]]
                cv2.imwrite(os.path.join(tiles_path, idx+"_"+input_filename+"_tile-{}-{}.png".format(val[0]['tile_ul'][0], val[0]['tile_ul'][1])), tile_image)

        # cv2.imwrite(os.path.join(bbox_tile_path, input_filename+"_tiles_bboxes.png"), image)

    return final_result

def prepare_dataset(dir_path, tif_filename, png_filename, json_filename, save_tiles, tiles_path, resize_image_width, resize_image_height, tile_width, tile_height):
        
    convert_tif_to_byte_gdal(os.path.join(dir_path, tif_filename), dir_path, ".png")
    
    bboxes = convert_geocoord_pixelcoord(os.path.join(dir_path, tif_filename), os.path.join(dir_path, json_filename))
    
    resize_image_opencv_pad(os.path.join(dir_path, png_filename), 
                            os.path.join(dir_path, png_filename), 
                            resize_image_width, 
                            resize_image_height, 
                            color=(255,255,255))
    
    bboxes_with_tiles               = split_image_to_tiles(os.path.join(dir_path, png_filename), 
                                                           bboxes, 
                                                           resize_image_width, 
                                                           resize_image_height, 
                                                           tile_width, 
                                                           tile_height, 
                                                           save_tiles, 
                                                           tiles_path)        
    bboxes_with_tiles_translated    = copy.deepcopy(bboxes_with_tiles)
    
    for idx, tile_bboxes in bboxes_with_tiles.items():
        if len(tile_bboxes[1])>0:
            bboxes_with_tiles_translated[idx]       = [0, []]
            translated_tile                         = { 'tile_ul': (0, 0), 
                                                        'tile_ur': (416, 0), 
                                                        'tile_br': (416, 416), 
                                                        'tile_bl': (0, 416)}
            bboxes_with_tiles_translated[idx][0]    = translated_tile
            for idx_bbox, bbox in enumerate(tile_bboxes[1]):
                original_tile   = tile_bboxes[0]
                translated_bbox = { 
                                    'bbox_ul': (bbox["bbox_ul"][0]-original_tile["tile_ul"][0], bbox["bbox_ul"][1]-original_tile["tile_ul"][1]), 
                                    'bbox_ur': (416-(original_tile["tile_ur"][0]-bbox["bbox_ur"][0]), bbox["bbox_ur"][1]-original_tile["tile_ur"][1]), 
                                    'bbox_br': (416-(original_tile["tile_br"][0]-bbox["bbox_br"][0]), 416-(original_tile["tile_br"][1]-bbox["bbox_br"][1])), 
                                    'bbox_bl': (bbox["bbox_bl"][0]-original_tile["tile_bl"][0], 416-(original_tile["tile_bl"][1]-bbox["bbox_bl"][1]))
                                    }
                bboxes_with_tiles_translated[idx][1].append(translated_bbox)
        else:
            del bboxes_with_tiles_translated[idx]
    
    return bboxes_with_tiles_translated

def preprocess_dataset_annotate(dir_path, tif_filename, png_filename, json_filename, save_tiles, tiles_path, resize_image_width, resize_image_height, tile_width, tile_height):

    bboxes_with_tiles_translated = prepare_dataset(dir_path, tif_filename, png_filename, json_filename, save_tiles, tiles_path, resize_image_width, resize_image_height, tile_width, tile_height)

    for root, dirs, files in os.walk(tiles_path):
        for file in files:
            for idx, new_tiles_bboxes in bboxes_with_tiles_translated.items():  
                if os.path.join(root, file).endswith((".png")) and file.startswith((idx+"_")) and tif_filename.split(".")[0] in file:
                    if len(new_tiles_bboxes[1])>0:
                        txt_path            = os.path.join(root, file.split(".")[0]+".txt")
                        yolo_file_object    = open(txt_path, "a")
                        image               = cv2.imread(os.path.join(root, file))
                        for idx_bbox, bbox in enumerate(new_tiles_bboxes[1]):
                   
                            bbox_ul = bbox["bbox_ul"]
                            bbox_br = bbox["bbox_br"]

                            bbox_ur = bbox["bbox_ur"]
                            bbox_bl = bbox["bbox_bl"]

                            bbox_width  = bbox_ur[0] - bbox_ul[0]
                            bbox_height = bbox_bl[1] - bbox_ul[1]

                            # Finding bbox midpoints
                            bbox_x_centre = (bbox_ul[0] + bbox_br[0])/2
                            bbox_y_centre = (bbox_ul[1] + bbox_br[1])/2

                            # Normalization
                            bbox_x_centre = bbox_x_centre / tile_width
                            bbox_y_centre = bbox_y_centre / tile_height
                            bbox_width    = bbox_width / tile_width
                            bbox_height   = bbox_height / tile_height

                            # Limiting upto fix number of decimal places
                            bbox_x_centre = format(bbox_x_centre, '.6f')
                            bbox_y_centre = format(bbox_y_centre, '.6f')
                            bbox_width    = format(bbox_width, '.6f')
                            bbox_height   = format(bbox_height, '.6f')
                            
                            yolo_file_object.write(f"{0} {bbox_x_centre} {bbox_y_centre} {bbox_width} {bbox_height}\n")
                            
                        yolo_file_object.close()
                        
                        cv2.imwrite(os.path.join(root, file), image)

def yolov3_train_dataset_preprocess(config, dataset_name):
    
    dataset_dirs =   [config["datasets"][dataset_name]["train_dir"]]

    for dataset_dir in dataset_dirs:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if os.path.join(root, file).endswith((".tif")):
                    tif_filename    = file
                    png_filename    = file.split(".")[0]+".png"
                    json_filename   = file.split(".")[0]+".json"
                    preprocess_dataset_annotate(dataset_dir, 
                                                tif_filename, 
                                                png_filename, 
                                                json_filename,
                                                config["datasets"][dataset_name]["save_tiles"],
                                                os.path.join(dataset_dir, config["datasets"][dataset_name]["tile_dir"]),
                                                resize_image_width=6240, 
                                                resize_image_height=6240, 
                                                tile_width=416, 
                                                tile_height=416)

        images = [image for image in sorted(os.listdir(os.path.join(dataset_dir, config["datasets"][dataset_name]["tile_dir"]))) if image[-4:]=='.png']
        annots = []
        for image in images:
            annot = image[:-4] + '.txt'
            annots.append(annot)
        images = pd.Series(images, name='image')
        annots = pd.Series(annots, name='text')
        df = pd.concat([images, annots], axis=1)
        df = pd.DataFrame(df)
        if "train" in dataset_dir:
            df.to_csv(os.path.join(os.path.dirname(dataset_dir),"train.csv"), index=False)

def yolov3_validation_dataset_preprocess(config, dataset_name):
    
    dataset_dirs =   [config["datasets"][dataset_name]["validation_dir"]]

    for dataset_dir in dataset_dirs:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if os.path.join(root, file).endswith((".tif")):
                    tif_filename    = file
                    png_filename    = file.split(".")[0]+".png"
                    json_filename   = file.split(".")[0]+".json"
                    preprocess_dataset_annotate(dataset_dir, 
                                                tif_filename, 
                                                png_filename, 
                                                json_filename,
                                                config["datasets"][dataset_name]["save_tiles"],
                                                os.path.join(dataset_dir, config["datasets"][dataset_name]["tile_dir"]),
                                                resize_image_width=6240, 
                                                resize_image_height=6240, 
                                                tile_width=416, 
                                                tile_height=416)

        images = [image for image in sorted(os.listdir(os.path.join(dataset_dir, config["datasets"][dataset_name]["tile_dir"]))) if image[-4:]=='.png']
        annots = []
        for image in images:
            annot = image[:-4] + '.txt'
            annots.append(annot)
        images = pd.Series(images, name='image')
        annots = pd.Series(annots, name='text')
        df = pd.concat([images, annots], axis=1)
        df = pd.DataFrame(df)
        if "validation" in dataset_dir:
            df.to_csv(os.path.join(os.path.dirname(dataset_dir),"validation.csv"), index=False)

def yolov3_test_dataset_preprocess(config, dataset_name):
    
    dataset_dirs =   [config["datasets"][dataset_name]["test_dir"]]

    for dataset_dir in dataset_dirs:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if os.path.join(root, file).endswith((".tif")):
                    tif_filename    = file
                    png_filename    = file.split(".")[0]+".png"
                    json_filename   = file.split(".")[0]+".json"
                    preprocess_dataset_annotate(dataset_dir, 
                                                tif_filename, 
                                                png_filename, 
                                                json_filename,
                                                config["datasets"][dataset_name]["save_tiles"],
                                                os.path.join(dataset_dir, config["datasets"][dataset_name]["tile_dir"]),
                                                resize_image_width=6240, 
                                                resize_image_height=6240, 
                                                tile_width=416, 
                                                tile_height=416)

if __name__=='__main__':
    yolov3_train_dataset_preprocess(config, "toppotsdam")
    yolov3_validation_dataset_preprocess(config, "toppotsdam")
    yolov3_test_dataset_preprocess(config, "toppotsdam")