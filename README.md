# Xcaliber-Task

This is a problem that was solved for the interview at xcaliber. The task was to perform semantic segmentation to detect cracks on the pavement. The dataset is used to train the model is available at 
[link](https://github.com/cuilimeng/CrackForest-dataset).
1. 	In order to train a model from scratch, download the dataset and store in  the filesystem and create a Anaconda Environment. The python version used for the project is 3.6
		
		$ conda create -n <env-name> --file requirements.txt

2. 	Activate the virtual environment
		
		$ source activate <env-name>

3. 	Run the [train.py](https://github.com/brijml/xcaliber-task/blob/master/train.py) file
		
		$ python train.py --basepath /home/brijml/Desktop/CrackForest-dataset-master --pretrained 0 --lr 1e-3 --savedir /home/brijml/Desktop/xcaliber-task/parameters/ --epoch 10

    where the basepath is the path to dataset, pretrained is flag indicating    whether to load a model to continue training(if it is set to 1 a "modelfile" parameter is used to give the path to the partially trained model), lr is the learning rate, savedir is the directory where the model is to be saved, epoch are the number of epochs to train the model for

4. 	After the model is trained you can evaluate the model using the evaluate.py which measures the average precision, recall and F1 score.
		
		$ python evaluate.py --basepath /home/brijml/Desktop/CrackForest-dataset-master --modelfile /home/brijml/Desktop/xcaliber-task/parameters/mymodel7.h5 --saveprediction /home/brijml/Desktop/xcaliber-task/output/
    where the basepath is the path to dataset, modelfile is the path to the partially model, saveprediction is where the prediction from the model is saved.

5.  You can also perform a data augmentation before training the model.
		
		$ python data_augmentation.py --basepath /home/brijml/Desktop/CrackForest-dataset-master
    where the basepath is the path to dataset.

## Evaluation Results
Both the models were trained with the same hyperparameters(i.e. equal number of epochs and learning rate) except one is trained with SGD optimizer and the other is trained with RMSProp optimizer. The model trained with RMSProp converged to the local optima faster than the one trained with SGD. An evalation can be performed with the Precision, Recall and F1 score. Calculating precision, recall and F1 score of each model,
| Model | Precision | Recall | F1 score
| ------ | ------ | ------ | ------ |
| with SGD| 0.6793339826220599 | 0.3867762951981743 | 0.4682298709208381 |
| with RMSProp | 0.7495153476508604 | 0.35647028274681797 | 0.46470270101566924 |

Higher value of precision shows that the model trained with RMSProp is more accurate is detecting cracks(since false postives are lesser in prediction) but lower value for recall for model trained with RMSProp shows there are higher false negatives in prediction which is also dependant on the threshold value(0.5 from [0-1] was used for experiments).

## Sample results
### First model(trained without batch normalisation)
![ex1](https://github.com/brijml/Object-Detection-Faster-RCNN/blob/fast_rcnn/examples/output1.png)
![ex2](https://github.com/brijml/Object-Detection-Faster-RCNN/blob/fast_rcnn/examples/output2.png)
![ex3](https://github.com/brijml/Object-Detection-Faster-RCNN/blob/fast_rcnn/examples/output3.png)

### Second model(trained with batch normalisation)
![ex1](https://github.com/brijml/Object-Detection-Faster-RCNN/blob/fast_rcnn/examples/output1.png)
![ex2](https://github.com/brijml/Object-Detection-Faster-RCNN/blob/fast_rcnn/examples/output2.png)
![ex3](https://github.com/brijml/Object-Detection-Faster-RCNN/blob/fast_rcnn/examples/output3.png)


### Conclusion
Training a neural network with RMSProp not only speeds up the training but also makes it smoother as compared to SGD opimizer which shows lot more oscillations in weight updates and loss values at each iteration. 
### References
[Zhun Fan,Yuming Wu, Jiewei Lu, and Wenji Li, “Automatic Pavement Crack Detection Based on Structured Prediction with the Convolutional Neural Network,” Arxiv,2018](https://arxiv.org/pdf/1802.02208.pdf)