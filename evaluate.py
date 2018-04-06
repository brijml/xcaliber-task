from keras.models import load_model
from data_utils import *
import os, argparse
import matplotlib
# matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default=1, help = 'path to the dataset directory')
	parser.add_argument('--modelfile', type=str, default="my-model.h5", help = 'path to load the model')
	parser.add_argument('--saveprediction', type=str, default="my-model.h5", help = 'path to load the model')
	return parser.parse_args()

if __name__ == '__main__':
	args = get_arguments()
	base_path = args.basepath 
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	image_files = os.listdir(image_path)
	test_images = get_test_files(image_path, gt_path)
	model = load_model(args.modelfile)

	for image_file in image_files:
		if image_file.endswith('.jpg'):
			img = mpimg.imread(os.path.join(image_path, image_file))
			img_expanded = np.expand_dims(img, axis=0)
			prediction = model.predict_on_batch(img_expanded)
			out = np.argmax(prediction[0], axis=2)
			m,n = out.shape
			out_channels = np.zeros((m,n,3), np.uint8)
			for i in range(3):
				out_channels[:,:,i] = out

			save_image = np.hstack([img, out_channels])
			show(img)
			show(prediction[0][:,:,0])
			show(prediction[0][:,:,1])
			show(out)
			# plt.imshow(save_image)
			# plt.savefig(os.path.join(args.saveprediction, image_file))