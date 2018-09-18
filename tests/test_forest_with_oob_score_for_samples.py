from context import bidir_submatrix
from bidir_submatrix import forest_with_oob_score_for_samples as Forest_oob
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification
from bidir_submatrix import helpers

import unittest


class Forest_with_oob_score_for_samples_TestCase(unittest.TestCase):
	def setUp(self):
		
		self.binary_problem_instance = make_classification(n_classes=2, n_samples=100, n_features=10, \
			n_informative=8, n_redundant=0,random_state=0, shuffle=False)
		self.Xb = self.binary_problem_instance[0]
		self.yb = self.binary_problem_instance[1]
		self.Xb, self.yb = helpers.binary_to_regression(self.Xb, self.yb)


		self.multiClass_problem_instance = make_classification(n_classes=4, n_samples=100, n_features=10, \
			n_informative=8, n_redundant=0,random_state=0, shuffle=False)
		self.Xmc = self.multiClass_problem_instance[0]
		self.ymc = self.multiClass_problem_instance[1]
		self.Xmc, self.ymc = helpers.multiClass_to_regression(self.Xmc, self.ymc, 4)


		self.multiLabel_problem_instance = make_multilabel_classification(n_classes=5, \
			n_labels=2, n_samples=100, n_features=10)
		self.Xml = self.multiLabel_problem_instance[0]
		self.yml = self.multiLabel_problem_instance[1]
		self.Xml, self.yml = helpers.multiLabel_to_regression(self.Xml, self.yml, 5)


	def test_oob_score_for_samples_shape_binary(self):
		self.RFR_oob_binary = Forest_oob.RandomForestRegressor(oob_score=True,
			oob_scores_samples=True)
		self.RFR_oob_binary.fit(self.Xb, self.yb)
		self.assertEqual(self.RFR_oob_binary.oob_scores_samples_.shape, (100, 1), 
			'Binary: Wrong shape for oob_score_for_samples produced by fit')


	def test_oob_score_for_samples_shape_multiClass(self):
		self.RFR_oob_multiClass = Forest_oob.RandomForestRegressor(oob_score=True,
			oob_scores_samples=True)
		self.RFR_oob_multiClass.fit(self.Xmc, self.ymc)
		self.assertEqual(self.RFR_oob_multiClass.oob_scores_samples_.shape, (100, 4), 
			'MultiClass: Wrong shape for oob_score_for_samples produced by fit')

	def test_oob_score_for_samples_shape_multiLabel(self):
		self.RFR_oob_multiLabel = Forest_oob.RandomForestRegressor(oob_score=True,
			oob_scores_samples=True)
		self.RFR_oob_multiLabel.fit(self.Xml, self.yml)
		self.assertEqual(self.RFR_oob_multiLabel.oob_scores_samples_.shape, (100, 5), 
			'MultiLabel: Wrong shape for oob_score_for_samples produced by fit')



if __name__ == '__main__':
	unittest.main(exit=False)
	