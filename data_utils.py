import numpy as np
import scipy.io as sio
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle
from keras.utils.np_utils import to_categorical


def show(image):
	if len(image.shape) == 2:
		plt.imshow(image, cmap='Greys')
	else:
		plt.imshow(image)
	plt.show()
	return

def rand_shuffle(files):
	for i in range(20):
		shuffle(files)

	return files

def validation_split(gt_files, ratio):
	no_train = int(len(gt_files)*(1-ratio))
	train_gt = gt_files[0:no_train]
	val_gt = gt_files[no_train:]
	return train_gt, val_gt

def get_test_files(image_path, gt_path):
	test_files = []
	image_files = os.listdir(image_path)
	gt_files = os.listdir(gt_path)
	gt_mod = [i.split('.')[0] for i in gt_files]
	for i, file_ in enumerate(image_files):
		if file_.split('.')[0] not in gt_mod:
			test_files.append(file_)

	return test_files

def load_gen(image_path, gt_path, gt_files, batch_size, n_classes):
	image_files = os.listdir(image_path)
	# gt_files = os.listdir(gt_path)
	gt_files = rand_shuffle(gt_files)
	for i in range(0,len(gt_files),batch_size):
		gt_files_batch = gt_files[i:i+batch_size] 
		images, labels = [], []
		for gt_file in gt_files_batch:
			image_file = gt_file.split('.')[0]+'.jpg'
			img = mpimg.imread(os.path.join(image_path, image_file))
			mat_contents = sio.loadmat(os.path.join(gt_path, gt_file))
			seg_mask = mat_contents['groundTruth'][0][0][0]
			seg_mask[seg_mask == 1] = 0
			seg_mask[seg_mask > 1] = 1
			images.append(img)
			labels.append(seg_mask) 
		
		categorical_labels = to_categorical(np.array(labels), num_classes=n_classes)
		yield np.array(images), categorical_labels

if __name__ == '__main__':
	base_path = '/home/brijml/Desktop/CrackForest-dataset-master'
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	gt_files = os.listdir(gt_path)
	gen_object = load_gen(image_path, gt_path, gt_files, 10, 2)
	for batch_train in gen_object:
		images = batch_train[0]
		labels = batch_train[1]
		for i,image in enumerate(images):
			show(image)
			show(labels[i][:,:,0])
			show(labels[i][:,:,1])
