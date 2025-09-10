import os
import cv2
import tensorflow as tf
from tensorflow import CenterCrop, ChannelShuffle, Rotate, CoarseDropout, HorisontalFlip
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn import train_test_split
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y


dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "test")
train_x, train_y = load_data(train_path)



print(f"Train:\t {len(train_x)} - {len(train_y)}")


create_dir("new_data/train/image/")
create_dir("new_data/train/mask/")
create_dir("new_data/test/image/")
create_dir("new_data/test/mask/")



H = 512
W = 512
i_s=100
for i in range(i_s):
	""" Extract the name """
	name = x.split("/")[-1].split(".")[0]
	save_path="new_data/train/"

	x = train_x[i]
	y = train_y[i]
	x = cv2.imread(path, cv2.IMREAD_COLOR)
	y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	aug = HorizontalFlip(p=1.0)
	augmented = aug(image=x, mask=y)
	x1 = augmented["image"]
	y1 = augmented["mask"]

	x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
	y2 = y

	aug = ChannelShuffle(p=1)
	augmented = aug(image=x, mask=y)
	x3 = augmented['image']
	y3 = augmented['mask']

	aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
	augmented = aug(image=x, mask=y)
	x4 = augmented['image']
	y4 = augmented['mask']

	aug = Rotate(limit=45, p=1.0)
	augmented = aug(image=x, mask=y)
	x5 = augmented["image"]
	y5 = augmented["mask"]

	X = [x, x1, x2, x3, x4, x5]
	Y = [y, y1, y2, y3, y4, y5]

	index = 0
	for i, m in zip(X, Y):
		
		crop = CenterCrop(H, W, p=1.0)
		augmented = crop(image=i, mask=m)
		i = augmented["image"]
		m = augmented["mask"]
		i = cv2.resize(i, (W, H))
		m = cv2.resize(m, (W, H))

		x_name = f"{name}_{index}.png"
		y_name = f"{name}_{index}.png"

		x_path = os.path.join(save_path, "image", x_name)
		y_path = os.path.join(save_path, "mask", y_name)

		cv2.imwrite(x_path, i)
		cv2.imwrite(y_path, m)


		index += 1
