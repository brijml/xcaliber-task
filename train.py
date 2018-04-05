import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop
from keras import regularizers

import os
from data_utils import load_gen

INPUT_SHAPE = (320,480,3)

def seg_model(n_classes):
	model = Sequential()
	model.add(Conv2D(16, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),
		input_shape=INPUT_SHAPE))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Conv2D(64, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#Convolutional Transposes
	model.add(Conv2DTranspose(64, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Conv2DTranspose(32, (3,3),
		strides=(2, 2), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Conv2DTranspose(16, (3,3),
		strides=(2, 2), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Conv2DTranspose(n_classes, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='softmax',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	return model


if __name__ == '__main__':
	base_path = '/home/brijml/Desktop/CrackForest-dataset-master'
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	model = seg_model(2)
	optimizer = RMSprop(lr=1e-3)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	model_save_dir = 'parameters/my-model.h5'

	while True:
		gen_object = load_gen(image_path, gt_path, 10, 2)
		for batch_train in gen_object:
			x,y = batch_train
			loss, accuracy = model.train_on_batch(x,y)
			print('loss:',loss,'accuracy:',accuracy)
		model.save(filepath)