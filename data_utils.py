import numpy as np
import scipy.io as sio
import os
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle, randint
# from keras.utils.np_utils import to_categorical


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

def get_shuffled(images,labels):
	no = len(images)
	range_no = list(range(no))
	for i in range(20):
		shuffle(range_no)

	image_temp,labels_temp = [],[]
	for i in range_no:
		image_temp.append(images[i])
		labels_temp.append(labels[i])

	return image_temp,labels_temp

def get_batch(image, seg_mask, ratio=3, h=13, s=2):
	indices_pos = np.where(seg_mask == 1)
	indices_neg = np.where(seg_mask == 0)
	# print(len(indices_pos[0]))
	patches, labels = [], []
	m,n = image.shape[:2]
	count, length = 0, len(indices_pos[0])
	# for i,ind in enumerate(indices_pos[0]):
	while count < 100:
		rand_int = randint(0, length)-1
		x, y = indices_pos[0][rand_int], indices_pos[1][rand_int]#
		# x,y = ind, indices_pos[1][i]
		if x-h-1 < 0 or y-h-1 < 0 or x+h+1 > m or y+h+1>n:
			continue
		else:
			assert(image[x-h:x+h+1, y-h:y+h+1].shape == (27,27,3))
			patches.append(image[x-h:x+h+1, y-h:y+h+1])
			labels.append(seg_mask[x-s:x+s+1, y-s:y+s+1].flatten())
			count+=1

	no_negs = ratio*len(patches)
	count, length = 0, len(indices_neg[0])
	while count<no_negs:
		rand_int = randint(0, length)-1
		x, y = indices_neg[0][rand_int], indices_neg[1][rand_int]
		if x-h-1 < 0 or y-h-1 < 0 or x+h+1 > m or y+h+1>n:
			continue
		else:
			assert(image[x-h:x+h+1, y-h:y+h+1].shape == (27,27,3))
			patches.append(image[x-h:x+h+1, y-h:y+h+1])
			labels.append(seg_mask[x-s:x+s+1, y-s:y+s+1].flatten())
			count+=1

	patches, labels = get_shuffled(patches, labels)
	return np.array(patches), np.array(labels)

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

def load_gen_v2(image_path, gt_path, gt_files):#, batch_size, n_classes):
	image_files = os.listdir(image_path)
	# gt_files = os.listdir(gt_path)
	gt_files = rand_shuffle(gt_files)
	for gt_file in gt_files:
		image_file = gt_file.split('.')[0]+'.jpg'
		img = mpimg.imread(os.path.join(image_path, image_file))
		mat_contents = sio.loadmat(os.path.join(gt_path, gt_file))
		seg_mask = mat_contents['groundTruth'][0][0][0]
		seg_mask[seg_mask == 1] = 0
		seg_mask[seg_mask > 1] = 1
		yield get_batch(img, seg_mask)


if __name__ == '__main__':
	base_path = '/home/brijml/Desktop/CrackForest-dataset-master'
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	gt_files = os.listdir(gt_path)
	gen_object = load_gen_v2(image_path, gt_path, gt_files)#, 10, 2)
	for batch_train in gen_object:
		images = batch_train[0]
		labels = batch_train[1]
		print(images.shape)
		# for i,image in enumerate(images):
			# show(image)
			# show(labels[i][:,:,0])
			# show(labels[i][:,:,1])
			# print(labels[i])
			