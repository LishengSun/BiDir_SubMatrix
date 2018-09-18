import numpy as np 
import matplotlib.pyplot as plt
import os

dataset = 'dorothea'
oob_scores_samples = np.loadtxt('../results/%s/oob_scores_samples.out'%dataset)

if len(oob_scores_samples.shape) == 1: # binary
	plt.plot(sorted(oob_scores_samples.tolist(), reverse=True), '.', label='label_1')

else: # multi**
	for cls_i in range(oob_scores_samples.shape[1]):
		oob_scores_samples_cls_i = oob_scores_samples[:, cls_i]
		plt.plot(sorted(oob_scores_samples_cls_i.tolist(), reverse=True), '.', label='label_%i'%cls_i)

plt.ylabel('oob score of samples in descending order')
plt.xlabel('sample')
plt.legend()
plt.title(dataset)
plt.savefig(os.path.join('../results_plots', dataset+'_oob_samples'))
plt.show()

