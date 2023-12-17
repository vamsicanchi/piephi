# Python Imports
import os
import sys
import time
import copy
import json
import math
import shutil
import random
from pprint import pprint

# Library Imports
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Gloabal Variable/Settings
sys.path.append("D:\\igis\\aiml\\core")

# Custom Imports
from appconfig import config
from library.utils.log import log


def run_inference_plots_modelwise(config, dataset_name, log):
    
    model_path      = os.path.dirname(config["datasets"][dataset_name]["save_checkpoint_file"])
    dir_path        = config["datasets"][dataset_name]["inference_dir"]

    models                          = []
    model_images                    = []
    model_confidences               = []
    
    for model_root, model_dirs, model_files in os.walk(model_path):
        for model_file in model_files:
            images                  = []
            confidences_per_epochs  = []
            if "epoch" in os.path.join(model_root, model_file):
                config["datasets"][dataset_name]["save_checkpoint_file"] = os.path.join(model_root, model_file)
                model_filename  = model_file.replace(".pth","")
                models.append(model_filename)
                for root, dirs, files in os.walk(dir_path):       
                    for file in files: 
                        if file.endswith(".geojson") and model_filename in file:
                            temp_confidences_per_epochs = []
                            images.append(file)
                            with open(os.path.join(root,file), 'r') as gjson:
                                geojson = json.load(gjson)
                            for feature in geojson["features"]:
                                temp_confidences_per_epochs.append(feature["properties"]["confidence_score"])
                            confidences_per_epochs.append(temp_confidences_per_epochs)
                model_images.append(images)
                model_confidences.append(confidences_per_epochs)

    # sns.set_theme(style='darkgrid', rc={'figure.dpi': 147}, font_scale=0.7)
    # fig, ax = plt.subplots(figsize=(7, 2))
                
    fig = go.Figure()
    
    subplot_rows = len(models) # /2 if len(models)%2==0 else math.ceil(len(models)/2)

    fig = make_subplots(rows=16, cols=1, subplot_titles=(models), row_heights=[6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]) # 

    for model_idx, model in enumerate(models):
        if len(model_images[model_idx])>0:
            data = []
            for idx, confidences in enumerate(model_confidences[model_idx]):
                for confidence in confidences:
                    data.append((model_images[model_idx][idx], confidence))

            # Create a DataFrame from the list of tuples
            df = pd.DataFrame(data, columns=['Image', 'Confidence'])

            # # Create the violin plot
            # sns.violinplot(x='Image', y='Confidence', data=df)
            # plt.xlabel('Image Name')
            # plt.ylabel('Confidence Score')
            # plt.title(model)
            # ax.set_title(model)
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            # plt.show()

            df['Image'] = df['Image'].str[12:16]+df['Image'].str[35:53]
            fig.append_trace(go.Violin(name=model, x=df['Image'], y=df['Confidence'], box_visible=True, meanline_visible=True), row=model_idx+1, col=1)

    fig.update_layout(title='Confidence Scores', autosize=True, width=1700, height=4500)
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.show()

def run_inference_plots_imagewise(config, dataset_name, log):
    
    model_path      = os.path.dirname(config["datasets"][dataset_name]["save_checkpoint_file"])
    image_path      = config["datasets"][dataset_name]["inference_dir"]

    models                  = [file.replace(".pth","") for file in os.listdir(model_path) if "epoch" in file]
    models_active           = []
    images                  = [file.replace(".tif","") for file in os.listdir(image_path) if file.endswith(".tif")]
    image_models_confidence = []
    
    for image in images:
        image_confidences = []
        for model in models:
            for root, dirs, files in os.walk(image_path):
                for file in files:
                    if file.endswith(".geojson") and model in file and image in file:
                        model_confidence = []
                        models_active.append(model)
                        with open(os.path.join(root,file), 'r') as gjson:
                            geojson = json.load(gjson)
                            for feature in geojson["features"]:
                                model_confidence.append(feature["properties"]["confidence_score"])
                        image_confidences.append(model_confidence)
        image_models_confidence.append(image_confidences)

    # print(len(images))
    # print(len(image_models_confidence))
    # for li in image_models_confidence:
    #     for i in li:
    #         print(len(list(set(models_active))),"----",len(li),"----",len(i))
    
    fig             = go.Figure()
    subplot_rows    = len(images)
    fig             = make_subplots(rows=subplot_rows, cols=1, subplot_titles=(images), row_heights=[6 for i in range(len(images))]) 

    colors = []
    for i in range(len(images)):
        colors.append('#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    
    models_active   = list(set(models_active)) 
    for image_idx, image_conf in enumerate(image_models_confidence):
        for model_idx, model_conf in enumerate(image_conf):
            data = []
            for conf in model_conf:
                data.append((models_active[model_idx], conf))

            # Create a DataFrame from the list of tuples
            df = pd.DataFrame(data, columns=['Model', 'Confidence'])
            df['Model'] = df['Model'].str[14:]
            fig.append_trace(go.Violin(name=images[image_idx][12:17]+"-"+df['Model'][model_idx], x=df['Model'], y=df['Confidence'], box_visible=True, meanline_visible=True, showlegend=True, legendgroup=models_active[model_idx],marker=dict(color=colors[image_idx]) ), row=image_idx+1, col=1)

    fig.update_layout(title='Confidence Scores', autosize=True, width=1700, height=9500, legend=dict(traceorder="normal", bgcolor='#E2E2E2', bordercolor="Black", borderwidth=2))
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.show()
    fig.write_html("image.html")

if __name__=='__main__':   
    # run_inference_plots_modelwise(config, "toppotsdam", log)
    run_inference_plots_imagewise(config, "toppotsdam", log)