# Homework 2

The model is built on the heart disease dataset. It is avalable on kaggle https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

To build and run the docker image you need to specify USER_ID and GROUP_ID, for example while in the directory run 
```
docker-compose build --build-arg USER_ID=1000 --build-arg GROUP_ID=1001 
docker-compose up
```

dockerfile - I'm using python 3.12.2 base just copying the folders and scripts and installing libraries from requriments.txt. Data permission configuration is also in the file.

docer-compose - I'm using volumes for the data and models folders 

The project has two folders:
* data - all the data for model training and making predictions. the folder has two files right noe:
  * heart.csv - data to used for model training
  * predict.csv - data we would make predictions on (this is just a sample of the heart.csv with the target dropped)
* models - all the models trained as txt files. can be read into lightgbm booster class. the folder has a few example models.

The project consists of two main scripts
* train.py - script to train the model. It splits the data into train/validation/test sets and trains a lightgbm model (uses mostly default parameters as adding any kind of regularization made test set performance worse)  after running the script it prints train and test set accuracy and also saves the model file in the models folder as model_{current time}.txt. to train the model on the desired dataset run
```
python train.py heart.csv # or another dataset. the dataset must be a csv file and be in the data folder
```
* predict.py - script to make predictions. the dataset must not have a target, it should be just Xs. it saves the predictions in the data folder as pred_{current time}.csv. you need to pass the model txt file and the dataset and the model file
```
python predict.py model.txt predict.csv # or any other model or data. the model must be in the models folder and the data - data folder
```

There is also a testing.ipynb file - just a jupyter notebook where I tested if the model actually worked

I changed the model from the pretrained CV model to the one I'm training myself on a small dataset, because installing torch and other libraries on the image took over 30 minutes each time, so It was very inconvinient for testing. Also images took up a lot of space
