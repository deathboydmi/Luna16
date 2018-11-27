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
        z = zip.ZipFile(path+zip_path)
        data_field.append(z.namelist())
        z.close()
    data_field = np.array(data_field, dtype='str')
    data_field = data_field.transpose()

    data_frame = pd.DataFrame(data_field, columns=[0, 1, 2])
    data_frame.to_csv("./data_frame.csv")
    return data_frame


#def load_model(name, path):
# z = zip.ZipFile("../data/subset0.zip")
# print(z.namelist())
# z.close()
# exit()
get_contain("../data/")