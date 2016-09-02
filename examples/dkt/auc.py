import numpy as np

f = file("pred_truth_array","rb")
pred_truth_array = np.load(f)
f.close()

print pred_truth_array.shape

from sklearn import metrics
y = pred_truth_array[:,1]
pred = pred_truth_array[:,0]
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print metrics.auc(fpr, tpr)