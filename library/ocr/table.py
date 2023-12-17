import os
import cv2
import warnings
import img2pdf
import pdf2image
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
poppler_path                          = 'D:\\poppler-0.68.0\\bin'
files                                 = ["Invoice_Sample_2.pdf"] 


def extract(file):
    
    if file.endswith((".png")):
        custom_oem_psm_config   = r'--oem 3 -c preserve_interword_spaces=1  --psm 3'
        image                   = Image.open(file)
        data                    = pytesseract.image_to_data(image, config=custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME, pandas_config=pytesseract.Output.DATAFRAME)
        # data.to_csv(file.replace(".pdf",".csv"))

    elif file.endswith((".pdf")):
        custom_oem_psm_config   = r'--oem 3 -c preserve_interword_spaces=1  --psm 3'
        image                   = pdf2image.convert_from_path(file, 500, poppler_path=poppler_path)
        opencvImage             = cv2.cvtColor(np.array(image[0]), cv2.COLOR_RGB2BGR)
        data                    = pytesseract.image_to_data(image[0], config=custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME, pandas_config=pytesseract.Output.DATAFRAME)
        data['centroid_x'] = 0
        data['centroid_y'] = 0
        n_boxes            = len(data['level'])
        for i in range(n_boxes):
            data['centroid_x'][i] = (data['left'][i]+data['left'][i]+data['width'][i])/2
            data['centroid_y'][i] = (data['top'][i]+data['top'][i]+data['height'][i])/2 
        
        data['centroid_tuple'] = data[['centroid_x', 'centroid_y']].apply(tuple, axis=1)
        data['table'] = False
        
        # tuple below is table column name and threhold
        table_columns               = [('DATE',101), ('ACTIVITY',110), ('DESCRIPTION',110), ('AMOUNT', 60)]
        table_columns_row_single    = {}
        table_columns_row_multiple  = {}

        for column in table_columns:
            df  = data[data['text'].fillna("NA").str.contains(column[0])]
            lis = list(df['centroid_tuple'].values)
            if len(lis)==1:
                table_columns_row_single[column[0]] = lis
            else:
                table_columns_row_multiple[column[0]] = lis

        for k1,v1 in table_columns_row_multiple.items():
            temp = []
            for centroid in v1:
                for k2,v2 in table_columns_row_single.items():
                    if v2[0][1]-centroid[1]<20:
                        temp.append(centroid)  
            table_columns_row_single[k1] = list(set(temp)) 
        
        amount_col      = []
        date_col        = []
        activity_col    = []
        description_col = []
         
        for column in table_columns:
            nearby_x_value = table_columns_row_single[column[0]][0][0]
            nearby_y_value = table_columns_row_single[column[0]][0][1]
            for i in range(n_boxes):
                if data['conf'][i]!=np.float64(-1.0):
                    if abs(data['centroid_x'][i]-nearby_x_value)<np.float64(column[1]):
                        data["table"][i] = True
                        if column[0]=='AMOUNT' and data['centroid_y'][i]>=nearby_y_value:
                            amount_col.append(list(data.iloc[i]))
                        elif column[0]=='DATE' and data['centroid_y'][i]>=nearby_y_value:
                            date_col.append(list(data.iloc[i]))
                        elif column[0]=='ACTIVITY' and data['centroid_y'][i]>=nearby_y_value:
                            activity_col.append(list(data.iloc[i]))
                        elif column[0]=='DESCRIPTION' and data['centroid_y'][i]>=nearby_y_value:
                            description_col.append(list(data.iloc[i]))

        data_table = data[data["table"]]
        
        data_table = data_table[['text','centroid_x','centroid_y','centroid_tuple']]

        
        print("Amount Column....")
        amount_df = pd.DataFrame(amount_col, columns=data.columns)
        amount_df = amount_df[['text','centroid_x','centroid_y','centroid_tuple']]
        print(amount_df)
 
        
        print("Date Column....")
        date_df = pd.DataFrame(date_col, columns=data.columns)
        date_df = date_df[['text','centroid_x','centroid_y','centroid_tuple']]
        print(date_df)

            
        print("Activity Column....")
        activity_df = pd.DataFrame(activity_col, columns=data.columns)
        activity_df = activity_df[['text','centroid_x','centroid_y','centroid_tuple']]
        print(activity_df)

            
        print("Description Column....")
        description_df = pd.DataFrame(description_col, columns=data.columns)
        description_df = description_df[['text','centroid_x','centroid_y','centroid_tuple']]
        print(description_df)
        
        
    for file in files:
        extract(file)