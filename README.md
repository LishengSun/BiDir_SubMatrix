Code for Bi-direction (feature / sample) selection.            

This is done by transforming all ML tasks to regression tasks (classification problems are transformed to predict the probability of the target class), and train a RandomForestRegressor to compute the feature importances and oob score for samples.

Please use the BiDir_SubMatrix_tutorial.ipynb as an end-to-end example.