"""
Author: Lisheng
Objective: demonstrate the idea of double selection
Steps:
(1) transform binary classification to regression: 2 outputs ~ 0/1
(2) Train RFR
(3) Recuparate the oob scores=mean-square-error(hard part)
"""
import numpy as np 
import sys
import os
import argparse

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from context import bidir_submatrix
from bidir_submatrix import forest_with_oob_score_for_samples as Forest_oob
from bidir_submatrix import helpers
import logging
import psutil
import traceback
import time
import platform



if __name__ == '__main__':

	######## set dirs ##########
	source_dir = '/Users/lishengsun/Dropbox/Meta-RL/BiDir_SubMatrix/AutoML1_data/'
	# source_dir = '/users/ao/lsun/Simulations/'
	
	data_root_dir = os.path.join(source_dir, 'entire_data')
	sys.path.insert(0, source_dir)
	results_dir = '/Users/lishengsun/Dropbox/Meta-RL/BiDir_SubMatrix/results'
	
	
	####### import from lib_dir############
	if platform.python_version().startswith('3.'):
		lib_dir = os.path.join(source_dir, 'lib3')
		sys.path.insert(0, lib_dir)
		from lib3.data_manager import DataManager #lib3 for python3
		from lib3 import data_io
	else:
		lib_dir = os.path.join(source_dir, 'lib')
		sys.path.insert(0, lib_dir)
		from lib.data_manager import DataManager
		from lib import data_io

	####### set args ############
	parser = argparse.ArgumentParser()
	parser.add_argument("--round", type=str, help='Round of the dataset')
	parser.add_argument("--dataset", type=str, help='basename of the dataset you want to test')
	parser.add_argument("--logging", type=str, help='True / False to generate log file for task. \
		Recommend True only when running on 1 dataset.')
	args = parser.parse_args()


	dataset = args.dataset
	round_ = args.round
	log = args.logging
	data_dir = os.path.join(data_root_dir, 'Round_'+round_)
	result_dir = os.path.join(results_dir, dataset)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)



	######## set logging ###########
	log_file = os.path.join(result_dir, dataset+'.log')
	logging.basicConfig(filename=log_file, level=logging.INFO, \
		format='%(asctime)s %(message)s', \
		filemode='w')
	logger = logging.getLogger()

	if log == 'False':
		logger.disabled = True
		os.remove(log_file)

	######### machine fingerprint ##########
	logger.info('Machine info: Platform: %s'%platform.platform())
	logger.info('Machine info: Memory: %s'%str(psutil.virtual_memory()))
	logger.info('Machine info: Python version: %s'%platform.python_version())
	######## run experiment ############
	
	t0 = time.time()
	logger.info('Dataset = %s'%(dataset))
	data = DataManager(input_dir=data_dir, basename=dataset)
	logger.info('Data loading completed: %s'%(str(data.info)))
	X, y = data.data['X_train'], data.data['Y_train']
	
	# transform to regression
	print (y[:5])
	if data.info['task'] != 'regression':
		logger.info('Transforming %s task to regression task'%(data.info['task']))
		if data.info['task'] == 'binary.classification':
			X, y = helpers.binary_to_regression(X, y)
		elif data.info['task'] == 'multiclass.classification':
			X, y = helpers.multiClass_to_regression(X, y, data.info['label_num'])
		elif data.info['task'] == 'multilabel.classification':
			X, y = helpers.multiLabel_to_regression(X, y, data.info['label_num'])
	print (y[:5])
	logger.info('Launching RandomForestRegressor to compute oob score for samples and feature importance.')
	# launch rf regressor
	rfr = Forest_oob.RandomForestRegressor(n_estimators=100, oob_score=True,
			oob_scores_samples=True)
	logger.info('Fitting RandomForestRegressor')
	rfr.fit(X, y)
	logger.info('Computing oob_scores_samples_ and feature_importances_')
	oob_scores_samples = rfr.oob_scores_samples_
	feature_importances = rfr.feature_importances_

	logger.info('Saving results to %s'%(result_dir))
	try:
		np.savetxt(os.path.join(result_dir, 'oob_scores_samples.out'), oob_scores_samples)
		np.savetxt(os.path.join(result_dir, 'feature_importances.out'), feature_importances)
		logging.info('Results saved.')
	except Exception as err:
		logger.error('Saving results failed: %s'%(traceback.print_tb(err.__traceback__)))

	logger.info('Total runtime = %f seconds.'%(time.time()-t0))







	# 