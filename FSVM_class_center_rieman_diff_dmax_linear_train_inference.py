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

from scipy.spatial.distance import cdist
from sklearn.utils.class_weight import compute_sample_weight

class FsvmCil:
    def __init__(self, kernel, decay='linear', beta=0.5, delta=1e-5, C=100):
        self.decay = decay
        self.beta = beta
        self.delta = delta
        self.C = C
        self.kernel = kernel
        self.model = None

    def assign_membership_values(self, X, y):

        d = np.zeros_like(y, dtype=float)

        d_max = np.zeros_like(y, dtype=float)

        X_neg = X[np.where(y == 0)].copy()
        X_pos = X[np.where(y == 1)].copy()

        d_neg = self.get_rieman_distance_from_mean(X_neg)
        d_pos = self.get_rieman_distance_from_mean(X_pos)

        d[y==0] = d_neg
        d[y==1] = d_pos

        d_max[y==0] = max(d_neg)
        d_max[y==1] = max(d_pos)

        #d_max = max(np.concatenate((d_neg, d_pos), axis=0))

        if self.decay == "exponential":
          self.memberships = 2/(1+np.exp(self.beta*d))

        if self.decay == "linear":
          self.memberships = 1 - d/(d_max + self.delta)

        sample_weights = np.ones_like(self.memberships)

        sample_weights[y == 0] = np.sum(y == 1)/np.sum(y == 0)

        self.memberships = self.memberships * sample_weights

        return self.memberships

    def fit(self, X, y):
        memberships = self.assign_membership_values(X, y)
        self.model = SVC(kernel=self.kernel, C=self.C)
        self.model.fit(X, y, sample_weight=memberships)

    def decision_function(self, X):
        if self.model is not None:
            return self.model.decision_function(X)
        else:
            raise Exception("Model is not fitted yet.")

    def get_rieman_distance_from_mean(self, X):
      Kij = self.kernel(X, X)
      return Kij.diagonal() - 2/len(X)*Kij.sum(axis=1) + 1/(len(X)**2)*Kij.sum()

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
    
### train with best params and predict
    
X_train = normalize(df_all_train, feature_cols, peaks_to_retain)
y_train = df_all_train['target'].values

X_test = normalize(df_test, feature_cols, peaks_to_retain)
y_test = df_test['target'].values


kernel = RBFKernel(2)
svm = FsvmCil(kernel, 'linear', delta=0.01, C=100)
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

with open('models/fsvm_class_center_rieman_diff_dmax.pkl','wb') as f:
    pickle.dump(svm, f)