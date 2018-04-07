import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dense, Flatten, Activation, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, SGD
from keras import regularizers
from logger import Logger

import os, argparse
from data_utils import *
from keras.layers.normalization import BatchNormalization
from keras import backend as K
K.set_learning_phase = 1

INPUT_SHAPE = (27,27,3)
	
def get_arguments():
	parser = argparse.ArgumentParser(description='Necessary variables.')
	parser.add_argument('--basepath', type=str, default=1, help = 'path to the dataset directory')
	parser.add_argument('--pretrained', type=int, default=1, help = 'Load pretrained model or not(1/0)')
	parser.add_argument('--batchsize', type=int, default=10, help = 'number of sample per batch')
	parser.add_argument('--modelfile', type=str, default="my-model.h5", help = 'path to be given when pretrained is set to 1')
	parser.add_argument('--lr', type=float, default=1e-4, help = 'learning_rate')
	parser.add_argument('--savedir', type=str, help = 'where the model is saved')
	parser.add_argument('--epoch', type=int, default=5, help = 'number of epochs')
	return parser.parse_args()

def seg_model():

	inps = Input(shape = INPUT_SHAPE)
	x = Conv2D(16, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation=None,
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(inps)
	x = BatchNormalization(axis=-1)(x, training=K.learning_phase())
	x = Activation('relu')(x)

	x = Conv2D(16, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation=None,
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization(axis=-1)(x, training=K.learning_phase())
	x = Activation('relu')(x)

	x = MaxPooling2D(pool_size=(2, 2))(x)

	x = Conv2D(32, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation=None,
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization(axis=-1)(x, training=K.learning_phase())
	x = Activation('relu')(x)

	x = Conv2D(32, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation=None,
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization(axis=-1)(x, training=K.learning_phase())
	x = Activation('relu')(x)

	x = MaxPooling2D(pool_size=(2, 2))(x)

	x = Flatten()(x)
	x = Dense(64, activation=None, kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization(axis=-1)(x, training=K.learning_phase())
	x = Activation('relu')(x)
	x = Dense(64, activation=None, kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization(axis=-1)(x, training=K.learning_phase())
	x = Activation('relu')(x)
	y = Dense(25, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)

	model = Model(inputs=inps, outputs=y)
	return model


if __name__ == '__main__':
	args = get_arguments()
	base_path = args.basepath 
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	gt_files = os.listdir(gt_path)
	train_gt, val_gt = validation_split(gt_files, 0.2)
	print("training on {} samples, validating on {} samples".format(len(train_gt), len(val_gt))) 

	save_dir = args.savedir
	epoch = args.epoch
	batch_obj = Logger('batch','batch.log','info')
	logger_batch = batch_obj.log()

	if args.pretrained == 0:
		model = seg_model()
		optimizer = RMSprop(lr=args.lr)
		# optimizer = SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=False)
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	else:
		model = load_model(args.modelfile)

	epoch_count = 0
	while epoch_count < epoch:
		#perform training
		batch_count = 0
		train_gen_object = load_gen_v2(image_path, gt_path, train_gt)#, args.batchsize, 2)
		for batch_train in train_gen_object:
			x,y = batch_train
			try:
				assert(x.shape[0] == y.shape[0])
				loss, accuracy = model.train_on_batch(x,y)
				logger_batch.info('training loss for epoch_no {} batch_number {} is loss:{}, accuracy:{}'.format(epoch_count, batch_count, loss, accuracy))
				batch_count+=1

			except Exception as e:
				print(e)
				continue

		#perform validation
		batch_count,total_loss = 0,0
		val_gen_object = load_gen_v2(image_path, gt_path, val_gt)#, args.batchsize, 2)
		for batch_val in val_gen_object:
			x,y = batch_val
			try:
				assert(x.shape[0] == y.shape[0])
				loss, accuracy = model.test_on_batch(x,y)
				batch_count+=1
				total_loss+=loss
			except Exception as e:
				print(e)
				continue

		logger_batch.info('validation loss for epoch_no {} is loss:{}, accuracy:{}'.format(epoch_count, (total_loss/batch_count), accuracy))

		filename = 'mymodel'+str(epoch_count)+'_bn.h5'
		model.save(os.path.join(save_dir, filename))
		epoch_count+=1