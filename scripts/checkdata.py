import os
import cv2 as cv
import numpy as np

DATA_PATH = "../GENERATED_DATA/"

if __name__ == "__main__":
	tree = os.walk(DATA_PATH)
	path_img_list = []
	for d, dirs, files in tree:
		for f in files:
			path = os.path.join(d, f)
			path_img_list.append(path)

path_img_list = list(filter(lambda x: x.endswith('.png'), path_img_list))
print(len(path_img_list))
for i, path_img in enumerate(path_img_list):
	if (i < 6100):
		continue
	img = cv.imread(path_img)
	try:
		cv.imshow("img", img)
	except cv.error:
		print(i, path_img)
		print("\t WILL BE DELETED")
		os.remove(path_img)
	else:
		cv.waitKey(1)