# Xcaliber-Task

This is a problem that was solved for the interview at xcaliber. The task was to perform semantic segmentation to detect cracks on the pavement. The dataset used to train the model is available at 
[link](https://github.com/cuilimeng/CrackForest-dataset).
1. 	In order to train a model from scratch, download the dataset and store in  the filesystem and create a Anaconda Environment(install Anaconda3 if it is not installed on your system). The python version used for the project is 3.6
		
		$ conda create -n <env-name> --file requirements.txt

2. 	Activate the virtual environment
		
		$ source activate <env-name>

3. 	Run the [train.py](https://github.com/brijml/xcaliber-task/blob/master/train.py) file
		
		$ python train.py --basepath <path to the dataset> --pretrained 0 --optimizer rms --lr 1e-3 --savedir <path where the model is to be saved> --epoch 10

    where the basepath is the path to dataset, pretrained is flag indicating whether to load a model to continue training(if it is set to 1 a "modelfile" parameter is used to give the path to the partially trained model), lr is the learning rate, savedir is the directory where the model is to be saved, epoch are the number of epochs to train the model for, optimizer is the optimizer to be used to train the model(rms/sgd). The decrease in the training loss and the validation loss is logged in the "batch.log" file created during training.

4. 	After the model is trained you can evaluate the model using the [evaluate.py](https://github.com/brijml/xcaliber-task/blob/master/evaluate.py) which measures the average precision, recall and F1 score.
		
		$ python evaluate.py --basepath <path to the dataset> --modelfile <path to the trained model> --saveprediction <path where the outputs are to be saved>

5.  You can also perform a data augmentation before training the model.
		
		$ python data_augmentation.py --basepath <path to the dataset>

6.  Alternatively you can also train the model with batch normalization layers.
		
		$ python train_with_bn.py --basepath <path to the dataset> --pretrained 0 --lr 1e-3 --savedir <path where the model is to be saved> --epoch 10


## Evaluation Results
Both the models were trained with the same hyperparameters(i.e. equal number of epochs and learning rate) except one is trained with SGD optimizer and the other is trained with RMSProp optimizer. The model trained with RMSProp converged to the local optima faster than the one trained with SGD. An evaluation of the models can be performed with the Precision, Recall and F1 score. Calculating precision, recall and F1 score of each model,

| Model | Precision | Recall | F1 score
| ------ | ------ | ------ | ------ |
| with SGD| 0.6793339826220599 | 0.3867762951981743 | 0.4682298709208381 |
| with RMSProp | 0.7495153476508604 | 0.35647028274681797 | 0.46470270101566924 |

Higher value of precision shows that the model trained with RMSProp is more accurate is detecting cracks(since false postives are lesser in prediction) but lower value for recall for model trained with RMSProp shows there are higher false negatives in prediction which is also dependant on the threshold value(0.5 from [0-1] was used for experiments).

## Sample results
### First model(trained with SGD)
![ex1](https://github.com/brijml/xcaliber-task/blob/master/output/trained_with_sgd/098.jpg)
![ex2](https://github.com/brijml/xcaliber-task/blob/master/output/trained_with_sgd/022.jpg)

### Second model(trained with RMSProp)
![ex1](https://github.com/brijml/xcaliber-task/blob/master/output/trained_with_RMS/069.jpg)
![ex2](https://github.com/brijml/xcaliber-task/blob/master/output/trained_with_RMS/073.jpg)


### Observation
Training a neural network with RMSProp not only speeds up the training but also makes it smoother as compared to SGD opimizer which shows lot more oscillations in weight updates and loss values at each iteration.
### References
[Zhun Fan,Yuming Wu, Jiewei Lu, and Wenji Li, “Automatic Pavement Crack Detection Based on Structured Prediction with the Convolutional Neural Network,” Arxiv,2018](https://arxiv.org/pdf/1802.02208.pdf)
