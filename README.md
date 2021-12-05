# Maching_learning_competition

## Predicting diamonds prices

![](https://i.pinimg.com/originals/95/4d/1b/954d1b6d95daf61e268775708c012909.gif)

# Objetive:

- Find the best machine learning model and params for a given dataset.
    - Kaggle competition: [Diamonds | datamad1021-rev](https://www.kaggle.com/c/diamonds-datamad1021-rev/overview)
    
# Method:

- Cleaning the dataset
- Find the best model that fits the data
- Train a model (fit & predict) with the data
- Submit the model tested in kaggle
- Win the comptition

# Structure:

* Folder **Data**
    - train.csv, test.csv and sample_submission.csv provided by Kaggle
    - modelo.csv (cleanead dataset)
    - submission.csv, submission_02.csv, submission_03.csv and submission_04.csv (differents submissions I have prepared)
    
* Folder **notebooks**: (jupyter notebooks)
    - 01_cleaning_studying.ipynb
    - 02_testing_model.ipynb
    
* Folder **src**:
    - traintest.py: functions we use during the training of our model
    
# Libraries

* [sys](https://docs.python.org/3/library/sys.html)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://seaborn.pydata.org/)
* [sklearn.model_selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* [metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
    - [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
* [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
    - [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    - [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    - [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso)
    - [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
    - [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
    - [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
* [numpy](https://numpy.org/)
* [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [ensemble](https://scikit-learn.org/stable/modules/ensemble.html/)
    - [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    - [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [svm](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [sklearn.tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [ensemble](https://scikit-learn.org/stable/modules/ensemble.html/)

