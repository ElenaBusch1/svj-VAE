import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from models import *
from root_to_numpy import *

# Example usage
num_elements = 16
element_size = 4
encoding_size = 32
nepochs=10
batchsize=32
#encoder = PermInvEncoder(num_elements, element_size, encoding_size)

# Input of shape (batch_size, num_elements, element_size)
#input_data = np.random.randn(500, num_elements, element_size).astype(np.float32)
bkg = read_vectors("../v6smallQCD.root", 1000)
sig = read_vectors("../user.ebusch.515502.root", 1000)

input_data = np.concatenate((bkg,sig),axis=0)

truth_bkg = np.zeros(bkg.shape[0])
truth_sig = np.ones(sig.shape[0])


truth_1D = np.concatenate((truth_bkg,truth_sig))
truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

print(input_data.shape, truth.shape)

# Encoded representation of shape (batch_size, encoding_size)
pfn = get_gvae([num_elements,element_size],encoding_size)

(X_train, X_test,
 Y_train, Y_test) = train_test_split(input_data, truth, test_size=0.1)

pfn.fit(X_train, Y_train,
        epochs=nepochs,
        batch_size=batchsize,
        verbose=1)

# get predictions on test data
preds = pfn.predict(X_test, batch_size=1000)

# get ROC curve
pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()
