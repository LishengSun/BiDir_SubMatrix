import unittest
from context import bidir_submatrix
from bidir_submatrix import helpers
from sklearn.datasets import make_classification
import numpy as np

class Helpers_TestCase(unittest.TestCase):

	def setUp(self):
		self.Xb = np.zeros((3, 3))
		self.yb1 = np.array([0,1,0])
		self.yb2 = np.array([[1,0], [0,1], [1,0]])
		
		self.Xmc = np.zeros((3, 3))
		self.ymc1 = np.array([0,2,1])
		self.ymc2 = np.array([[1,0,0], \
							[0,0,1], \
							[0,1,0]])

		self.yml = np.array([[1, 0, 1],
							[0, 0, 1],
							[1, 1, 0]])


		# self.multiClass_instance = make_classification(n_classes=4, n_samples=100, n_features=10, \
		# 	n_informative=8, n_redundant=0,random_state=0, shuffle=False)
		# self.Xmc = self.multiClass_instance[0]
		# self.ymc = self.multiClass_instance[1]
		

	def assertArrayEqual(self, arr1, arr2):
		self.assertTrue(np.alltrue(arr1 == arr2))

	def test_validate_problem_instance(self):
		self.assertTrue(helpers.validate_problem_instance(self.Xb, self.yb1))
		self.assertTrue(helpers.validate_problem_instance(self.Xmc, self.ymc1))

		self.assertFalse(helpers.validate_problem_instance(self.Xb[:-1,:], self.yb1))
		self.assertFalse(helpers.validate_problem_instance(self.Xmc[:-1,:], self.ymc1))


	def test_binary_to_regression(self):
		y_reg1 = helpers.binary_to_regression(self.Xb, self.yb1)[1]
		y_reg2 = helpers.binary_to_regression(self.Xb, self.yb2)[1]
		self.assertArrayEqual(y_reg1, np.array([0,1,0]))
		self.assertArrayEqual(y_reg2, np.array([0,1,0]))

	def test_multiClass_to_regression(self):
		y_reg1 = helpers.multiClass_to_regression(self.Xmc, self.ymc1, 3)[1]
		y_reg2 = helpers.multiClass_to_regression(self.Xmc, self.ymc2, 3)[1]
		self.assertArrayEqual(y_reg1, np.array([[1,0,0], \
										[0,0,1], \
										[0,1,0]]))
		self.assertArrayEqual(y_reg2, np.array([[1,0,0], \
										[0,0,1], \
										[0,1,0]]))

	def test_multiLabel_to_regression(self):
		y_reg = helpers.multiLabel_to_regression(self.Xmc, self.yml, 3)[1]
		self.assertArrayEqual(y_reg, np.array([[1, 0, 1],
										[0, 0, 1],
										[1, 1, 0]]))


if __name__ == '__main__':
	unittest.main(exit=False)




