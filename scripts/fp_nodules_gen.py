import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.utils import shuffle

import os
import zipfile as zip

import nodules_gen as ng

if (os.name == 'nt'):
	DATA_PATH = "E:\\DL\\datasets\\Luna16\\dataset\\DATA\\"
	CSV_PATH = "E:\\DL\\datasets\\Luna16\\dataset\\CSVFILES\\"
	EXT_FILES = "E:\\DL\\datasets\\Luna16\\dataset\\EXTFILES\\"
	NEW_DATA_PATH = "E:\\DL\\datasets\\Luna16\\dataset\\GENERATED_DATA\\NODULES\\"
else:
	DATA_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/DATA/"
	CSV_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/CSVFILES/"
	EXT_FILES = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/EXTFILES/"
	NEW_DATA_PATH = "/media/deathboydmi/HDDeathBoyDmi/DL/datasets/Luna16/dataset/GENERATED_DATA/FP_NODULES/"

GEN_ING_SIZE = 128

def load_candidates(csv_path, num):
	labels = pd.read_csv(csv_path)
	labels['seriesuid'] = labels['seriesuid'].astype('str')
	labels['coordX'] = labels['coordX'].astype('float')
	labels['coordY'] = labels['coordY'].astype('float')
	labels['coordZ'] = labels['coordZ'].astype('float')
	labels = shuffle(labels)
	labels = labels[:num]
	labels.sort_index(inplace=True)
	data_ndarray = labels.as_matrix()
	return labels, data_ndarray

def get_info(row):
	name = row['seriesuid'] + ".mhd"
	x, y, z, label = row['coordX'], row['coordY'], row['coordZ'], row['class']
	return name, x, y, z, label

def show_metaimg(path, voxelCoord_list=[]):
	numpyImage, numpyOrigin, numpySpacing = ng.load_itk_image(path)
	Z, X, Y = numpyImage.shape
	cv.namedWindow(path, cv.WINDOW_NORMAL)
	def nothing(x):
		pass
	cv.createTrackbar('Z', path, 0, Z-1, nothing)
	z = 0
	for voxelCoord in voxelCoord_list:
		numpyImage = markNodule(numpyImage, voxelCoord)
	while (True):
		slice_model = ng.normalizePlanes(numpyImage[z,:,:])
		cv.imshow(path, slice_model)
		k = cv.waitKey(1) & 0xFF
		if (k==113):
			break
		z = cv.getTrackbarPos('Z', path)

def markNodule(img, voxelCoord):
	voxelWidth = 128
	top_left = (int(voxelCoord[2] - voxelWidth / 2), int(voxelCoord[1] - voxelWidth / 2))
	bottom_right = (int(voxelCoord[2] + voxelWidth / 2), int(voxelCoord[1] + voxelWidth / 2))
	marked_img = cv.rectangle(img[int(voxelCoord[0]),:,:], top_left, bottom_right, 255, 1)
	img[int(voxelCoord[0]),:,:] = marked_img
	return img

def getNodulesInfo(name, data_ndarray, numpyOrigin, numpySpacing):
	ind, _ = np.where(data_ndarray==name[:len(name)-4])
	worldCoords = []
	for i in ind:
		worldCoords.append(data_ndarray[i,1:5])
	worldCoords = [np.fliplr(worldCoords)[i].astype(float) for i in range(len(ind))]
	voxelCoords = []
	for worldCoord in worldCoords:
		voxelCoord = ng.worldToVoxelCoord(worldCoord[1:], numpyOrigin, numpySpacing)
		voxelCoord = np.append(voxelCoord, worldCoord[0])
		voxelCoords.append(voxelCoord)

	return voxelCoords

#-------------------------------------------------------------------------------------------------------
def data_generation(numpyImage, voxelCoords, newImgSize, vis=False, save_path=""):
	new_data = []
	ind = 0
	for voxelCoord in voxelCoords:
		Z = int(voxelCoord[0])
		Y = int(voxelCoord[2])
		X = int(voxelCoord[1])

		top = X - newImgSize//2
		left = Y - newImgSize//2
		if (top < 0):
			top = 0
		if (left < 0):
			left = 0
		bottom = top + newImgSize
		right = left + newImgSize
		if (bottom > 511):
			bottom = 511
			top = bottom - newImgSize
		if (right > 511):
			right = 511
			left = right - newImgSize

		new_img = numpyImage[Z, top:bottom, left:right]
		new_img = ng.normalizePlanes(new_img)

		if(vis==True):
			cv.imshow("img", new_img)
			cv.waitKey(0)
		if(save_path!=""):
			cv.normalize(new_img, new_img, 0, 255, cv.NORM_MINMAX)
			cv.imwrite(save_path+"__FP__"+str(ind)+".png", new_img)
		else:
			new_data.append(new_img)
		ind += 1
	return new_data
#-------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	df, list_data = ng.get_data_content(DATA_PATH, CSV_PATH)
	print(df.head())
	print(list_data.shape)
	labels, list_annotation = load_candidates(CSV_PATH+"candidates_V2.csv", 11000)
	print(labels.head())
	print(list_annotation.shape)
	last_file = ""
	generated_nodules = []
	ind = 0
	for i, row in labels.iterrows():
		name, x, y, z, label = get_info(row)
		if (name == last_file):
			continue
		mhd_file, raw_file = ng.extract_model(name, list_data, DATA_PATH, EXT_FILES)
		numpyImage, numpyOrigin, numpySpacing = ng.load_itk_image(mhd_file)
		voxelCoords = getNodulesInfo(name, list_annotation, numpyOrigin, numpySpacing)
		#ng.show_metaimg(mhd_file, voxelCoords)
		print(ind,'/',list_annotation.shape[0], name[:len(name)-4], len(voxelCoords))
		data_generation(numpyImage, voxelCoords, GEN_ING_SIZE, save_path=NEW_DATA_PATH+name[:len(name)-4])
		last_file = name
		os.remove(mhd_file)
		os.remove(raw_file)
		ind += 1

	print("exit")
	exit()
