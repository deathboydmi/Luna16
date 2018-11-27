import numpy as np
import pandas as pd
import cv2 as cv
import SimpleITK as itk

import os
import zipfile as zip


def load_candidates(path):
    labels = pd.read_csv(path)
    labels = pd.read_csv(path)
    labels['seriesuid'] = labels['seriesuid'].astype('str')
    labels['label'] = labels['label'].astype('bool')
    labels.head()
    return labels

def get_info(row):
    path = "./data/" + row['seriesuid'] + ".mhd"
    x, y, z = row['coordX'], row['coordY'], row['coordZ']
    return path, x, y, z

def get_contain(path):
    listdir = os.listdir(path)
    data_field = []
    for zip_path in listdir:
        if (zip.is_zipfile(path+zip_path)):
            z = zip.ZipFile(path+zip_path)
            name_list = z.namelist()
            if (len(name_list) == 176):
                name_list.append('none')
                name_list.append('none')
            data_field.append(name_list)
            z.close()

    data_field = np.asmatrix(data_field, dtype=str)
    print(data_field.shape)
    data_field = data_field.transpose()
    print(data_field.shape)

    data_frame = pd.DataFrame(data_field, columns=["subset0","subset1","subset2","subset3","subset4",
                                                "subset5","subset6","subset7","subset8","subset9"])
    data_frame.to_csv("./data_frame.csv")
    return data_frame


#def load_model(name, path):


df = get_contain("/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/data/")
