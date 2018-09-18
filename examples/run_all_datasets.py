import os
import datetime
import time
import logging
import platform, psutil



if __name__ == '__main__':
	
	# round_datasets = {'0': ['adult', 'cadata', 'digits', 'dorotha', 'newsgroups'],\
					# '1': ['christine', 'philippine', 'jasmine', 'madeline', 'sylvine'], \
					# '2': ['volkert', 'albert', 'dilbert', 'robert', 'fabert']}

	round_datasets = {'3': ['alexis', 'dionis', 'grigoris', 'jannis', 'wallis'], \
					'4': ['evita', 'flora', 'helena', 'tania', 'yolanda'], \
					'5': ['arturo', 'carlo', 'waldo', 'marco', 'pablo']}
	######## set logging ###########
	logging.basicConfig(filename='run_all_datasets.log', level=logging.INFO, \
		format='%(asctime)s %(message)s', \
		filemode='w')
	logger = logging.getLogger()

	######### machine fingerprint ##########
	logger.info('Machine info: Platform: %s'%platform.platform())
	logger.info('Machine info: Memory: %s'%str(psutil.virtual_memory()))
	logger.info('Machine info: Python version: %s'%platform.python_version())
	
	t0 = time.time()
	for rnd in round_datasets.keys():
		for ds in round_datasets[rnd]:
			logger.info('Now Start ======%s======'%(ds))
			os.system('python sample/demo.py --round %s --dataset %s --logging False'%(rnd, ds))
	logger.info('Total runtime = %f seconds.'%(time.time()-t0))








