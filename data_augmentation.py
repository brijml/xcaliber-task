import cv2
import numpy as np
import random
from shutil import copyfile
import os, argparse

def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default=1, help = 'path to the dataset directory')
	return parser.parse_args()

def save_images(rand_ints, operation):
	for int_ in rand_ints:
		image_file = image_files[int_]
		if image_file.split('.')[0] not in gt_mod:
			continue

		img = cv2.imread(os.path.join(image_path, image_file))

		if operation == 'apply_noise':
			noise = cv2.randn(img,(0),(20))
			img = img + noise

		elif operation == 'blur':
			img = cv2.blur(img,(5,5))

		elif operation == 'average':
			kernel = np.ones((5,5),np.float32)/25
			img = cv2.filter2D(img,-1,kernel)

		cv2.imwrite(os.path.join(image_path, image_file.split('.')[0]+'_alt.jpg'),img)
		src = os.path.join(gt_path,image_file.split('.')[0]+'.mat')
		dst = os.path.join(gt_path,image_file.split('.')[0]+'_alt.mat')
		copyfile(src, dst)

	return

if __name__ == '__main__':
	args = get_arguments()
	base_path = args.basepath 
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	gt_files = os.listdir(gt_path)
	gt_mod = [i.split('.')[0] for i in gt_files]
	image_files = os.listdir(image_path)
	rand_ints = random.sample(range(1,len(image_files)), int(0.1*len(image_files)))
	save_images(rand_ints, 'blur')
	rand_ints = random.sample(range(1,len(image_files)), int(0.1*len(image_files)))
	save_images(rand_ints, 'average')