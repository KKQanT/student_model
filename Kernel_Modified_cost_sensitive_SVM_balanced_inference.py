import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

feature_cols = [f'x{i}' for i in range(30)]

peaks_to_retain = [5, 7, 9, 17, 18, 25, 29, 3, 4, 6, 11, 13, 15, 19, 21, 23, 26, 27, 28]
peaks_to_retain = [index - 1 for index in peaks_to_retain]

import scipy.io
mat = scipy.io.loadmat('data/full_spectra_feature_set_b12g.mat')

#################################################################################

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

def calculate_gmean(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn+fp)
  sensitivity = tp / (tp+fn)

  print('sensitivity: ', sensitivity)
  print('specificity: ', specificity)

  return np.sqrt(specificity * sensitivity)

def calculate_se_sp(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn+fp)
  sensitivity = tp / (tp+fn)

  return sensitivity, specificity

def calculate_se_sp_gmeans_with_threshold_(y_true, y_pred, T=0):
  y_pred_class = np.zeros(len(y_pred))
  y_pred_class[y_pred > T] = 1
  se, sp = calculate_se_sp(y_true, y_pred_class)

  return {"se": se,
          "sp": sp,
          "g_mean": np.sqrt(se * sp),
          'T':T
          }

def calculate_partial_auc(y_true, y_scores, sensitivity_threshold=0.8):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    idx = np.where(tpr > sensitivity_threshold)

    fpr_partial = fpr[idx]
    tpr_partial = tpr[idx]

    try:

      partial_auc = auc(fpr_partial, tpr_partial)

    except:
      print('auc error')
      partial_auc = 0

    return partial_auc

def evaluate(y_true, y_pred, T=0, se=0.8):
  result = calculate_se_sp_gmeans_with_threshold_(y_true, y_pred, T)
  partial_auc = calculate_partial_auc(y_true, y_pred, se)
  result['partial_auc'] = partial_auc
  return result

df_test = pd.DataFrame({
    "patient_id": mat['batchg_pat_id'].flatten(),
    "target": mat['batchg_class_id'].flatten(), #<-will be processed later
    "class_id": mat['batchg_class_id'].flatten()
})
df_test_features = pd.DataFrame(mat['batchg_feat'], columns=feature_cols)
df_test = pd.concat((df_test, df_test_features), axis=1)
df_test['id'] = df_test.index
df_test.loc[df_test['target'] != 1, 'target'] = 0
df_test = pd.DataFrame(df_test[df_test['class_id'] != 3]).reset_index(drop=True)
df_test = df_test.fillna(0)

def normalize(df, feature_cols, selected_cols):
  features = df[feature_cols].values
  features_norm = np.sqrt(np.sum(features**2, axis=1))
  features_test = features / features_norm[:, np.newaxis]
  return features_test[:, selected_cols]

from sklearn.gaussian_process.kernels import Kernel, RBF
import numpy as np
from scipy.spatial.distance import cdist

class RBFKernel(Kernel):

    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
        K = np.exp(-self.gamma * pairwise_sq_dists)
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True
    
X_test = normalize(df_test, feature_cols, peaks_to_retain)
y_test = df_test['target'].values

######################## Load model ######################
import pickle
with open('models/kernel_modified.pkl', 'rb') as f:
    svm = pickle.load(f)

tuned_threshold = 1.1912385544924478

y_pred = svm.decision_function(X_test)

final_result = evaluate(y_test, y_pred, tuned_threshold)
print("--------------------------final_result-----------------------------")
print(final_result)
