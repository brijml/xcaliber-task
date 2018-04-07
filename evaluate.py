from keras.models import load_model
from data_utils import *
import os, argparse
import matplotlib
# matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
K.set_learning_phase(0)

def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default=1, help = 'path to the dataset directory')
	parser.add_argument('--modelfile', type=str, default="my-model.h5", help = 'path to load the model')
	parser.add_argument('--saveprediction', type=str, default="my-model.h5", help = 'path to load the model')
	return parser.parse_args()

def get_batches(image, h=13):
	m,n,p = image.shape
	patches = []
	for i in range(0,m):
		for j in range(0,n):
			if i-h-1 < 0 or j-h-1 < 0 or i+h+1 > m or j+h+1>n:
				continue
			else:
				patches.append(image[i-h:i+h+1, j-h:j+h+1])

	return np.array(patches)

def infer(prediction, h=13, s=2):
	out = np.zeros((m,n), np.float32)
	count = 0
	for i in range(0,m):
		for j in range(0,n):
			if i-h-1 < 0 or j-h-1 < 0 or i+h+1 > m or j+h+1>n:
				continue
			else:
				out[i-s:i+s+1,j-s:j+s+1] += prediction[count].reshape((5,5))
				count+=1

	return out[h:m-h,h:n-h]

def apply_zeropad(img, h=13):
	m,n,p = img.shape 
	out = np.zeros((m+2*h,n+2*h,p), np.uint8)
	out[h:h+m,h:h+n] = img
	return out

def save_output(img, out_norm):
	out_channels = np.zeros_like(img)
	for i in range(3):
		out_channels[:,:,i] = out_norm*255

	save_image = np.hstack([img, out_channels])
	plt.subplot()
	plt.imshow(save_image)
	plt.savefig(os.path.join(args.saveprediction, image_file))
	return

def calc_precision_recall(ytrue, ypred):
	true_pos = len(np.where(np.logical_and(ytrue==ypred, ytrue==1))[0])
	true_neg = len(np.where(np.logical_and(ytrue==ypred, ytrue==0))[0])
	false_pos = len(np.where(np.logical_and(ytrue!=ypred, ytrue==1))[0])
	false_neg = len(np.where(np.logical_and(ytrue!=ypred, ytrue==0))[0])
	precision = float(true_pos)/(true_pos+false_pos)
	recall = float(true_pos)/(true_pos+false_neg)
	F1 = float(2*precision*recall)/(precision+recall)
	return precision, recall, F1

if __name__ == '__main__':
	args = get_arguments()
	base_path = args.basepath 
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	image_files = os.listdir(image_path)
	test_images = get_test_files(image_path, gt_path)
	gt_files = os.listdir(gt_path)
	train_gt, val_gt = validation_split(gt_files, 0.2)
	model = load_model(args.modelfile)
	
	# for l in model.layers:
	# 	l.trainable = False

	tot_p, tot_r, tot_f = 0, 0, 0
	count = 0
	for gt_file in val_gt:
		#load the label
		mat_contents = sio.loadmat(os.path.join(gt_path, gt_file))
		seg_mask = mat_contents['groundTruth'][0][0][0]
		seg_mask[seg_mask == 1] = 0
		seg_mask[seg_mask > 1] = 1

		#load the image
		image_file = gt_file.split('.')[0] + '.jpg'
		img = mpimg.imread(os.path.join(image_path, image_file))
		img1 = apply_zeropad(img)
		m,n,p = img1.shape
		batch = get_batches(img1)
		try:
			a = 5000
			total_prediction = np.zeros((batch.shape[0], 25))
			for i in range(0,len(batch),a):
				prediction = model.predict_on_batch(batch[i:i+a])
				total_prediction[i:i+a] = prediction
		except Exception as e:
			print(e)
			continue

		out = infer(total_prediction)
		out_norm = np.zeros_like(out)
		out_norm = cv2.normalize(out, out_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		
		# show(out_norm)
		out_norm[out_norm>0.5] = 1
		out_norm[out_norm<0.5] = 0
		# show(out_norm)

		#save the output
		save_output(img, out_norm)
		precision, recall, F1 = calc_precision_recall(seg_mask, out_norm)
		tot_p+=precision 
		tot_r+=recall
		tot_f+=F1
		count+=1
		print(tot_p, tot_r, tot_f)
	print("average precision:{}, average_recall:{}, average_F1_score{}".format(tot_p/count, tot_r/count, tot_f/count))

#Evaluation results for two models.
#average precision:0.7076627951050088, average_recall:0.35894027909052256, average_F1_score0.4610767084769732
#average precision:0.6793339826220599, average_recall:0.3867762951981743, average_F1_score0.4682298709208381