Code for Bi-direction (feature / sample) selection.            

This is done by transforming all ML tasks to regression tasks, 
and train a RandomForestRegressor to compute the feature importances and 
oob score for samples.