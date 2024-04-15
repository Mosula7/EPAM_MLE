# The Project consists of two parts:
* model_optimization
  * optimize_model.py - takes two optinal command line keyword arguments -d - data to optimize the model on (data.csv as default) -n number of trials for the 5 fold corss validation hyperparameter tunning. I'm using optima to optimze a lightgbm model. For each iteration and each cross validation step I'm logging accuracy and auc into mlflow (churn_random_search_cv experimet). after the trials are done a model is trained on the best hyperparameters, model accuracy and auc is evaluated on the train, validation and test sets, this gets logged into mlflow, including the model file (churn_best experiment) and the model booster object is saved as a txt into a models subdirectory.
  * models - models trained with the best hyperparameters
  * mlflow.db - local database to track all models
  * data - data to optimize the model on. IMPORTANT: the data we want to optimize the model on needs to be in this directory
  * dockerfile - builds a docker image and also hosts a mlflow server on port 5000
  * requrements.txt - python libraries required
  * to run the experiments in the container simply run
  ```
    python optimize_model.py
  ```
* model_deploymnet
  * app.py - hosts a fast api server using uvicorn on port 8000 where you can make predictions using a model in the models directory. after going to local host 8000 add /docs at the end and you will see predict post where you can pass model features and it will return a churn probability.
  * customer.py class for storing data required for the model
  * dockerfile - builds an image and locally hosts a fast api server.
  * models - directory to store models in.
 
* docker-compose.yml - builds model_deployment and model_optimization images. also makes volumes for data, models and mlflow.db in model_optimization and models in model_deployment. to build images and run containers while in the module3 directory run
```
docker-compose up
```
* testing.ipynb - notebook used to test the process.


