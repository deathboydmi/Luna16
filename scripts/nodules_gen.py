import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
import zipfile as zip

DATA_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/DATA/"
CSV_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/CSVFILES/"
EXT_FILES = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/EXTFILES/"

if (os.name == 'nt'):
	DATA_PATH = "E:\\DL\\datasets\\Luna16\\dataset\\DATA\\"
	CSV_PATH = "E:\\DL\\datasets\\Luna16\\dataset\\CSVFILES\\"
	EXT_FILES = "E:\\DL\\datasets\\Luna16\\dataset\\EXTFILES\\"
else:
	DATA_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/DATA/"
	CSV_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/CSVFILES/"
	EXT_FILES = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/EXTFILES/"


def load_candidates(csv_path):
	labels = pd.read_csv(csv_path)
	labels = pd.read_csv(csv_path)
	print(labels.head())
	labels['seriesuid'] = labels['seriesuid'].astype('str')
	labels['label'] = labels['diameter_mm'].astype('str')
	return labels

def get_info(row):
	name = row['seriesuid'] + ".mhd"
	x, y, z = row['coordX'], row['coordY'], row['coordZ']
	return name, x, y, z

def get_data_content(data_path, csv_path):
	listdir = os.listdir(data_path)
	data_field = []
	for zip_path in listdir:
		if (zip.is_zipfile(data_path+zip_path)):
			z = zip.ZipFile(data_path+zip_path)
			name_list = z.namelist()
			name_list = list(map(lambda x: x[8:], name_list))
			name_list = list(filter(lambda x: x[-1] != 'w', name_list))
			if (len(name_list) == 88):
				name_list.append('none')
			data_field.append(name_list)
			z.close()

	data_field = np.asmatrix(data_field, dtype=str)
	data_frame = data_field.transpose()
	data_frame = pd.DataFrame(data_frame, columns=["subset0","subset1","subset2","subset3","subset4",
											"subset5","subset6","subset7","subset8","subset9"], dtype=str)
	data_frame.to_csv(csv_path+"content.csv")
	return data_frame, data_field


def extract_model(name, content, data_path, ext_path):
	num_sub, _ = np.where(content == name)
	path = data_path + "subset" + str(int(num_sub))
	zip_path = path + ".zip"
	path = "subset" + str(int(num_sub)) + "/"
	print(path)
	mhd_file = path + name
	raw_file = mhd_file.replace(".mhd", ".raw")
	print(raw_file)
	if (zip.is_zipfile(zip_path)):
		z = zip.ZipFile(zip_path)
		z.extract(raw_file, path=ext_path)
		z.extract(mhd_file, path=ext_path)
		z.close()

	return (ext_path+mhd_file), (ext_path+raw_file)

def load_itk_image(filename):
	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)

	numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
	numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

	return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacvoxelCoording):
	stretchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = stretchedVoxelCoord / spacing
	return voxelCoord

def normalizePlanes(npzarray):
	 
	maxHU = 400.
	minHU = -1000.
 
	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray>1] = 1.
	npzarray[npzarray<0] = 0.
	return npzarray

def show_metaimg(path):
	numpyImage, numpyOrigin, numpySpacing = load_itk_image(path)
	print (numpyImage.shape, numpyOrigin, numpySpacing)
	cv.namedWindow(path, cv.WINDOW_NORMAL)
	def nothing(x):
		pass
	cv.createTrackbar('Z', path, 0, 511, nothing)
	z = 0
	while (True):
		slice_model = normalizePlanes(numpyImage[:,z,:])
		cv.imshow(path, slice_model)
		k = cv.waitKey(1) & 0xFF
		if (k==113):
			break
		z = cv.getTrackbarPos('Z', path)


# labels = load_candidates(CSV_PATH+"annotations.csv")
# for i, row in labels.iterrows():
# 	name, x, y, z = get_info(row)
# 	mhd_file, raw_file = extract_model(name, list_data, DATA_PATH, EXT_FILES)
# 	numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_file)
# 	voxelCoord = worldToVoxelCoord(np.asarray([float(z),float(y),float(x)]), numpyOrigin, numpySpacing)

# 	data_augmentation(numpyImage, voxelCoord)

labels = load_candidates(CSV_PATH+"annotations.csv")
df, list_data = get_data_content(DATA_PATH, CSV_PATH)
print(df.head())
name = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.mhd"
mhd_file, raw_file = extract_model(name, list_data, DATA_PATH, EXT_FILES)
print(mhd_file, raw_file)
show_metaimg(mhd_file)

# exit()

# df, list_data = get_data_content(DATA_PATH, CSV_PATH)
# name = "1.3.6.1.4.1.14519.5.2.1.6279.6001.231002159523969307155990628066.mhd"
# mhd_file, raw_file = extract_model(name, list_data, DATA_PATH, EXT_FILES)
# print(mhd_file, raw_file)

# print(numpyImage.shape)
# print(numpyOrigin)
# print(numpySpacing)
