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

###############################################################################

import pandas as pd

df = pd.DataFrame({
    "patient_id": mat['batch2_pat_id'].flatten(),
    "target": mat['batch2_class_id'].flatten(), #<-will be processed later
    "class_id": mat['batch2_class_id'].flatten()
})
df_features = pd.DataFrame(mat['batch2_feat'], columns=feature_cols)
df = pd.concat((df, df_features), axis=1)
df['id'] = df.index
df.loc[df['target'] != 1, 'target'] = 0
df_patient = df.groupby(['patient_id']).mean().reset_index()[['patient_id', "target"]]
df_patient.loc[df_patient['target'] > 0, 'target'] = 1
df = pd.DataFrame(df[df['class_id'] != 3]).reset_index(drop=True)
df_all_train = df.fillna(0).copy()

################################################################################

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
    
X_train = normalize(df_all_train, feature_cols, peaks_to_retain)
y_train = df_all_train['target'].values

X_test = normalize(df_test, feature_cols, peaks_to_retain)
y_test = df_test['target'].values

from afc_imbalanced_learning.afc import AFSCTSvm

kernel = RBFKernel(0.5)
svm = AFSCTSvm(
    C=0.01,
    class_weight="balanced",
    neg_eta=1 * 1,
    pos_eta=1,
    kernel=kernel,
    ignore_outlier_svs=True
)

svm.fit(X_train, y_train)

y_train_pred = svm.decision_function(X_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
best_T = thresholds[np.argmax(tpr >= 0.8)]
train_results = evaluate(y_train, y_train_pred, best_T)

y_pred = svm.decision_function(X_test)

final_result = evaluate(y_test, y_pred, best_T)
print("--------------------------final_result-----------------------------")
print(final_result)

import pickle

with open('models/kernel_modified.pkl','wb') as f:
    pickle.dump(svm, f)

### save results as requests
import os

y_pred_label = np.where(y_pred > best_T, 1, 0)

model_name = "Kernel_Modified_cost_sensitive_SVM_balanced"
results_dir = f"results/{model_name}"

os.makedirs(results_dir, exist_ok=True)

pd.DataFrame(y_pred_label, columns=["labels"]).to_csv(f"{results_dir}/labels.csv", index=False)
pd.DataFrame(y_pred, columns=["scores"]).to_csv(f"{results_dir}/scores.csv", index=False)