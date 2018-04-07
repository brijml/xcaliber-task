# Xcaliber-Task

This is a problem that was solved for the interview at xcaliber. The task was to perform semantic segmentation to detect cracks on the pavement. The dataset is used to train the model is available at https://github.com/cuilimeng/CrackForest-dataset.
1
In order to train a model from scratch, download the dataset and store in the filesystem and create a Anaconda Environment. The python version used for the project is 3.6
	$ conda create -n <env-name> --file requirements.txt

Run the train.py file
	$ python train.py --basepath /home/brijml/Desktop/CrackForest-dataset-master --pretrained 0 --lr 1e-3 --savedir /home/brijml/Desktop/xcaliber-task/parameters/ --epoch 10

where the basepath is the path to dataset, pretrained is flag indicating whether to load a model to continue training(if it is set to 1 a "modelfile" parameter is used to give the path to the partially trained model), lr is the learning rate, savedir is the directory where the model is to be saved, epoch are the number of epochs to train the model for

After the model is trained you can evaluate the model using the evaluate.py which measures the average precision, recall and F1 score.
	$ python evaluate.py --basepath /home/brijml/Desktop/CrackForest-dataset-master --modelfile /home/brijml/Desktop/xcaliber-task/parameters/mymodel7.h5 --saveprediction /home/brijml/Desktop/xcaliber-task/output/

where the basepath is the path to dataset, modelfile is the path to the partially model, saveprediction is where the prediction from the model is saved.

You can also do a data augmentation before beginning the training to increase the number of images.
	$ python data_augmentation.py --basepath /home/brijml/Desktop/CrackForest-dataset-master

where the basepath is the path to dataset.

# Sample results
