"""
No need to add l441-443 in /Users/lishengsun/anaconda/lib/python2.7/site-packages/sklearn/ensemble/forest.py
already commented out
Now the question is how to compute oob scores on clf.estimators_[0].predict_proba(X[unsampled_indices, :])
Check l425....
"""


from sklearn.ensemble2 import RandomForestClassifier
from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier(max_depth=2, random_state=0, oob_score=True, samples_oob=True, n_estimators=100)
clf.fit(X,y)

# import sklearn.ensemble2.forest
# unsampled_indices = sklearn.ensemble2.forest._generate_unsampled_indices(clf.estimators_[0].random_state,1000)
# pred_un0 = clf.estimators_[0].predict_proba(X[unsampled_indices, :])


### (1) need to re-write _set_oob_score() line425, no need to unsampled.. again
### (2) change the source code, save l425-... in real time