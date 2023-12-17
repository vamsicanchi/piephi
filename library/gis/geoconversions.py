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

# Library Imports
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import rasterio
import rasterio as rio
from rasterio import windows
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.transform import rowcol, xy
from rasterio.warp import reproject, calculate_default_transform, Resampling
import pyproj
from pyproj import Transformer
# import geopandas as gpd
import shapely
from shapely.geometry import box

# Custom Imports

# Gloabal Variable/Settings
warnings.filterwarnings("ignore")
gdal.SetConfigOption('CPL_LOG', 'NUL')

class CRSTransformer():
    
    """Transforms map points in some CRS into pixel coordinates.

    Each transformer is associated with a particular RasterSource.
    """

    def __init__(self,
                 transform=None,
                 image_crs: Optional[str] = None,
                 map_crs: Optional[str] = None):
        self.transform = transform
        self.image_crs = image_crs
        self.map_crs = map_crs

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates (eg. lon/lat). x and y can be
            single values or array-like.

        Returns:
            (x, y) tuple in pixel coordinates
        """
        pass

    def pixel_to_map(self, pixel_point):
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates. x and y can be
            single values or array-like.

        Returns:
            (x, y) tuple in map coordinates (eg. lon/lat)
        """
        pass

    def get_image_crs(self):
        return self.image_crs

    def get_map_crs(self):
        return self.map_crs

    def get_affine_transform(self):
        return self.transform

class RasterioCRSTransformer(CRSTransformer):
    """Transformer for a RasterioRasterSource."""

    def __init__(self, transform, image_crs, map_crs='epsg:4326'):
        """Constructor.

        Args:
            transform: Rasterio affine transform
            image_crs: CRS of image in format that PyProj can handle eg. wkt or init
                string
            map_crs: CRS of the labels
        """
        self.map2image = Transformer.from_crs(map_crs, image_crs, always_xy=True)
        self.image2map = Transformer.from_crs(image_crs, map_crs, always_xy=True)
        super().__init__(transform, image_crs, map_crs)

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        image_point = self.map2image.transform(*map_point)
        pixel_point = rowcol(self.transform, image_point[0], image_point[1])
        pixel_point = (pixel_point[1], pixel_point[0])
        return pixel_point

    def pixel_to_map(self, pixel_point):
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in map coordinates
        """
        image_point = xy(self.transform, int(pixel_point[1]),
                         int(pixel_point[0]))
        map_point = self.image2map.transform(*image_point)
        return map_point

    @classmethod
    def from_dataset(cls, dataset, map_crs='epsg:4326'):
        # if dataset.crs is None:
        #     return IdentityCRSTransformer()
        transform = dataset.transform
        image_crs = dataset.crs.wkt
        return cls(transform, image_crs, map_crs)

def convert_geocoord_pixelcoord(image_path, geojson_path):
    
    with open(geojson_path,'r') as json_file:
        geojson_data = json.load(json_file)

    polygons = []

    for attribute,values in geojson_data.items():
        if attribute=="features":
            for feature in values:
                polygon_coords = feature["geometry"]["coordinates"][0][:4]
                polygon_coords = [tuple(i) for i in polygon_coords]
                polygons.append(polygon_coords)

    bboxes = {}

    with rasterio.open(image_path) as src:
        for idx_polygon, polygon in enumerate(polygons):
            temp = {}
            for idx_coord, coord in enumerate(polygon):
                geocoord_to_imgcoord = RasterioCRSTransformer(src.transform, src.crs.wkt)
                pixel_point = geocoord_to_imgcoord.map_to_pixel(coord)
                if idx_coord==0:
                    temp["bbox_ul"] = pixel_point
                if idx_coord==2:
                    temp["bbox_br"] = pixel_point
                if idx_coord==1:
                    temp["bbox_ur"] = pixel_point
                if idx_coord==3:
                    temp["bbox_bl"] = pixel_point
            bboxes.update({str(idx_polygon): temp})

    return bboxes

def convert_geojson_yoloformat(image_path, geojson_path, class_id):
    
    yolo_dir            = os.path.dirname(geojson_path)
    yolo_file           = os.path.basename(geojson_path).split(".")[0]+".txt"
    yolo_path           = os.path.join(yolo_dir, yolo_file)
    yolo_file_object    = open(yolo_path, "a")

    with open(geojson_path,'r') as json_file:
        geojson_data = json.load(json_file)

    image = cv2.imread(image_path)
    image_height, image_width, image_channels = image.shape 

    polygons = []

    for attribute,values in geojson_data.items():
        if attribute=="features":
            for feature in values:
                polygon_coords = feature["geometry"]["coordinates"][0][:4]
                polygon_coords = [tuple(i) for i in polygon_coords]
                polygons.append(polygon_coords)

    polygon_rects = []

    with rasterio.open(image_path) as src:
        for idx_polygon, polygon in enumerate(polygons):
            temp = {}
            for idx_coord, coord in enumerate(polygon):
                geocoord_to_imgcoord = RasterioCRSTransformer(src.transform, src.crs.wkt)
                pixel_point = geocoord_to_imgcoord.map_to_pixel(coord)
                if idx_coord==0:
                    temp["bbox_ul"] = pixel_point
                if idx_coord==2:
                    temp["bbox_br"] = pixel_point
                if idx_coord==1:
                    temp["bbox_ur"] = pixel_point
                if idx_coord==3:
                    temp["bbox_bl"] = pixel_point
            polygon_rects.append(temp)

    for polygon in polygon_rects:
        
        bbox_ul = polygon["bbox_ul"]
        bbox_br = polygon["bbox_br"]

        bbox_ur = polygon["bbox_ur"]
        bbox_bl = polygon["bbox_bl"]

        bbox_width  = bbox_ur[0] - bbox_ul[0]
        bbox_height = bbox_bl[1] - bbox_ul[1]

        # Finding bbox midpoints
        bbox_x_centre = (bbox_ul[0] + bbox_br[0])/2
        bbox_y_centre = (bbox_ul[1] + bbox_br[1])/2

        # Normalization
        bbox_x_centre = bbox_x_centre / image_width
        bbox_y_centre = bbox_y_centre / image_height
        bbox_width    = bbox_width / image_width
        bbox_height   = bbox_height / image_height

        # Limiting upto fix number of decimal places
        bbox_x_centre = format(bbox_x_centre, '.6f')
        bbox_y_centre = format(bbox_y_centre, '.6f')
        bbox_width    = format(bbox_width, '.6f')
        bbox_height   = format(bbox_height, '.6f')
        
        yolo_file_object.write(f"{class_id} {bbox_x_centre} {bbox_y_centre} {bbox_width} {bbox_height}\n")

    yolo_file_object.close()

def convert_tif_to_byte_gdal(image_path, output_path, image_ext):
    
    if image_ext in [".jpg", ".jpeg"]:
        jpg_file        = os.path.join(output_path, os.path.basename(image_path).split(".")[0]+image_ext)
        gdal_tif_img    = gdal.Open(image_path)
        jpeg_driver     = gdal.GetDriverByName("JPEG")
        jpeg_driver.CreateCopy(jpg_file, gdal_tif_img, strict=0)

    if image_ext in [".png"]:
        png_file        = os.path.join(output_path, os.path.basename(image_path).split(".")[0]+image_ext)
        gdal_tif_img    = gdal.Open(image_path)
        png_driver      = gdal.GetDriverByName("PNG")
        png_driver.CreateCopy(png_file, gdal_tif_img, strict=0)

def resize_image_rasterio(input_image, output_image, resize_to_width, resize_to_height):

    """Reproject a file to match the shape and projection of existing raster. 

    Parameters
    ----------
    input_image : (string) path to input file to resize
    output_image : (string) path to output file tif
    """
    # open input
    with rasterio.open(input_image) as src:
     
        # calculate the output transform matrix
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,     # input CRS
            src.crs, 
            resize_to_width,   # input width
            resize_to_height,  # input height 
            *src.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
        )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": src.crs,
                            "transform": dst_transform,
                            "width": dst_width,
                            "height": dst_height,
                            "nodata": 0})
        # print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(output_image, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest)

def resize_all_rasterio(input_path, output_path, resize_to_width, resize_to_height, allowed_image_extensions):

    # get all the pictures in directory
    images = []

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(tuple(allowed_image_extensions)):
                images.append(os.path.join(root, file))
    
    for image in images:

        filename_ext = os.path.basename(image).split(".")

        # open input
        with rasterio.open(image) as src:

            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                                                                                src.crs,     # input CRS
                                                                                src.crs, 
                                                                                resize_to_width,   # input width
                                                                                resize_to_height,  # input height 
                                                                                *src.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                                                                              )

            # set properties for output
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({"crs": src.crs,
                                "transform": dst_transform,
                                "width": dst_width,
                                "height": dst_height,
                                "nodata": 0})
            # print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
            # open output
            with rasterio.open(os.path.join(output_path, filename_ext[0]+"_resized."+filename_ext[1]), "w", **dst_kwargs) as dst:
                # iterate through bands and write using reproject function
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest)

def split_image_gdal(input_image, output_path, output_image, tile_width=416, tile_height=416, fill_value=255):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ds = gdal.Open(input_image)

    filename = os.path.basename(input_image).split(".")[0]

    band = ds.GetRasterBand(1)

    xsize, ysize                = (band.XSize, band.YSize)
    tile_size_x, tile_size_y    = tile_width, tile_height
    
    tile_list = {}

    complete_x = xsize // tile_size_x
    complete_y = ysize // tile_size_y

    residue_x = xsize % tile_size_x
    residue_y = ysize % tile_size_y

    # for part A
    for j in range(complete_y):
        for i in range(complete_x):
        
            Xmin = i * tile_size_x
            Xmax = i * tile_size_x + tile_size_x - 1
            Ymin = j * tile_size_y
            Ymax = j * tile_size_y + tile_size_y - 1
            # do patch creation here
            com_string = "gdal_translate -of GTIFF -srcwin " + str(Xmin) + ", " + str(Ymin) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + \
                          str(input_image)+ " " +os.path.join(str(output_path), str(filename)+"-"+str(output_image.format(int(Xmin), int(Ymin))))
            os.system(com_string)

    x_residue_count = 1
    
    y_residue_count = 1

    # for part B
    for j in range(complete_y):
        Xmin = tile_size_x * complete_x
        Xmax = tile_size_x * complete_x + residue_x - 1
        Ymin = j * tile_size_y
        Ymax = j * tile_size_y + tile_size_y - 1
        # do patch creation here
        com_string = "gdal_translate -of GTIFF -srcwin " + str(Xmin) + ", " + str(Ymin) + ", " + str(residue_x) + ", " + str(tile_size_y) + " " + \
                      str(input_image)+ " " +os.path.join(str(output_path), str(filename)+"-"+str(output_image.format(int(Xmin), int(Ymin))))
        os.system(com_string)

    # for part C
    for i in range(complete_x):
        Xmin = i * tile_size_x
        Xmax = i * tile_size_x + tile_size_x - 1
        Ymin = tile_size_y * complete_y
        Ymax = tile_size_y * complete_y + residue_y - 1
        com_string = "gdal_translate -of GTIFF -srcwin " + str(Xmin) + ", " + str(Ymin) + ", " + str(tile_size_x) + ", " + str(residue_y) + " " + \
                      str(input_image)+" "+os.path.join(str(output_path), str(filename)+"-"+str(output_image.format(int(Xmin), int(Ymin))))
        os.system(com_string)

    # for part D
    Xmin = complete_x * tile_size_x
    Ymin = complete_y * tile_size_y
    com_string = "gdal_translate -of GTIFF -srcwin " + str(Xmin) + ", " + str(Ymin) + ", " + str(residue_x) + ", " + str(residue_y) + " " + \
                  str(input_image)+ " " +os.path.join(str(output_path), str(filename)+"-"+str(output_image.format(int(Xmin), int(Ymin))))
    os.system(com_string)

def split_image_rasterio(input_image_path, output_path, output_image, width_offset=(False,0), height_offset=(False,0), tile_width=416, tile_height=416, fill_value=255):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def split_tiles(ds, tile_width=416, tile_height=416):
        
        offsets = product(range(0, ds.meta['width'], tile_width), range(0, ds.meta['height'], tile_height)) 
        
        for row_off, col_off in offsets:
            
            if width_offset[0]:
                row_off = row_off+width_offset[1]
            if height_offset[0]:
                col_off = col_off+height_offset[1]

            window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height).intersection(window)
            transform = windows.transform(window, ds.transform)
            
            yield window, transform
    
    filename = os.path.basename(input_image_path).split(".")[0]
    
    with rio.open(input_image_path) as ds:
        
        meta = ds.meta.copy()
        meta['driver']='PNG'
        for window, transform in split_tiles(ds, tile_width, tile_height):
            
            meta['transform']               = transform
            meta['width'], meta['height']   = window.width, window.height
            
            outpath = os.path.join(output_path, str(filename)+"-"+output_image.format(int(window.col_off), int(window.row_off))+".png")
            
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(ds.read(boundless=True, window=window, fill_value=fill_value))

def plot_raster_geojson(config, input_image_path, geojson_path, save_as_png, output_image_path, log):
    
    with open(geojson_path,'r') as json_file:
        geojson_data = json.load(json_file)

    image = cv2.imread(input_image_path)

    polygons = []

    for attribute,values in geojson_data.items():
        if attribute=="features":
            for feature in values:
                polygon_coords = feature["geometry"]["coordinates"][0][:4]
                polygon_coords = [tuple(i) for i in polygon_coords]
                polygons.append(polygon_coords)

    polygon_rects = []

    with rasterio.open(input_image_path) as src:
        for polygon in polygons:
            temp = {}
            for idx, coord in enumerate(polygon):
                geocoord_to_imgcoord = RasterioCRSTransformer(src.transform, src.crs.wkt)
                pixel_point = geocoord_to_imgcoord.map_to_pixel(coord)
                if idx==0:
                    temp["ul"] = pixel_point
                if idx==2:
                    temp["br"] = pixel_point
                if idx==1:
                    temp["ur"] = pixel_point
                if idx==3:
                    temp["bl"] = pixel_point
            polygon_rects.append(temp)

    for polygon in polygon_rects:
        
        ul = polygon["ul"]
        br = polygon["br"]

        ur = polygon["ur"]
        bl = polygon["bl"]

        bbox_width  = ur[0] - ul[0]
        bbox_height = bl[1] - ul[1]

        # Finding bbox midpoints
        bbox_x_centre = (ul[0] + br[0])/2
        bbox_y_centre = (ul[1] + br[1])/2

        # Draw a rectangle with green line borders of thickness of 2 px
        image = cv2.rectangle(image, ul, br, config["opencv"]["bbox_color"], config["opencv"]["bbox_thickness"])
  
    # Displaying the image 
    cv2.imshow(config["opencv"]["window_name"], image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_as_png:
        cv2.imwrite(output_image_path, image)

def plot_all_bboxes(config, input_image, json_path, output_image):
    
    with open(json_path,'r') as json_file:
        json_data = json.load(json_file)

    image = cv2.imread(input_image)
    
    image_height, image_width, image_channels = image.shape 

    polygons = []

    for attribute,values in json_data.items():
        if attribute=="features":
            for feature in values:
                polygon_coords = feature["geometry"]["coordinates"][0][:4]
                polygon_coords = [tuple(i) for i in polygon_coords]
                polygons.append(polygon_coords)

    polygon_rects = []

    with rasterio.open(input_image) as src:
        for polygon in polygons:
            temp = {}
            for idx, coord in enumerate(polygon):
                geocoord_to_imgcoord = RasterioCRSTransformer(src.transform, src.crs.wkt)
                pixel_point = geocoord_to_imgcoord.map_to_pixel(coord)
                if idx==0:
                    temp["ul"] = pixel_point
                if idx==2:
                    temp["br"] = pixel_point
                if idx==1:
                    temp["ur"] = pixel_point
                if idx==3:
                    temp["bl"] = pixel_point
            polygon_rects.append(temp)

    for polygon in polygon_rects:
        
        ul = polygon["ul"]
        br = polygon["br"]

        ur = polygon["ur"]
        bl = polygon["bl"]

        bbox_width  = ur[0] - ul[0]
        bbox_height = bl[1] - ul[1]

        # Finding bbox midpoints
        bbox_x_centre = (ul[0] + br[0])/2
        bbox_y_centre = (ul[1] + br[1])/2

        # Draw a rectangle with green line borders of thickness of 2 px
        image = cv2.rectangle(image, ul, br, config.BBOX_COLOR, config.BBOX_THICKNESS)
  
    # Displaying the image 
    cv2.imshow(config.WINDOW_NAME, image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_image, image)