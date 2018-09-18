from sklearn.preprocessing import LabelBinarizer
import numpy as np

def validate_problem_instance(X, y):
	"""
	check if X.shape[0] == y.shape[0]
	"""
	if X.shape[0] == y.shape[0]:
		return True
	else:
		# raise ValueError('X.shape[0] and y.shape[0] must be equal')
		return False

def binary_to_regression(X, y):
	if validate_problem_instance:
		if len(y.shape) == 1: #not one-hot-encoded yet
			return X, y
		elif y.shape == (X.shape[0], 2): #if y is predict_proba
			y = np.argmax(y, axis=1)
			return X, y
		else: 
			raise ValueError('Bad shape of y')
	else:
		raise ValueError('X.shape[0] and y.shape[0] must be equal')

def multiClass_to_regression(X, y, num_class):
	if validate_problem_instance:
		if len(y.shape) == 1: #not one-hot-encoded yet
			enc = LabelBinarizer()
			y = enc.fit_transform(y)
			return X, y
		elif y.shape[1] == num_class:
			return X, y
		else:
			raise ValueError('Cannot understand the shape of y, please check it')
	else:
		raise ValueError('X.shape[0] and y.shape[0] must be equal')


def multiLabel_to_regression(X, y, num_label):
	if validate_problem_instance:
		if y.shape[1] == num_label: 
			return X, y
		else:
			raise ValueError('Cannot understand the shape of y, please check it')
	else:
		raise ValueError('X.shape[0] and y.shape[0] must be equal')

