import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from models import *
from models_archive import *
from root_to_numpy import *
from plot_helper import *

# Example usage
num_elements = 16
element_size = 4
encoding_size = 32
phi_dim = 128
nepochs=10
batchsize=100
#encoder = PermInvEncoder(num_elements, element_size, encoding_size)

# Input of shape (batch_size, num_elements, element_size)
#input_data = np.random.randn(500, num_elements, element_size).astype(np.float32)
bkg = read_vectors("../v6smallQCD.root", 10000)
sig = read_vectors("../user.ebusch.515502.root", 10000)

input_data = np.concatenate((bkg,sig),axis=0)

truth_bkg = np.zeros(bkg.shape[0])
truth_sig = np.ones(sig.shape[0])

truth_1D = np.concatenate((truth_bkg,truth_sig))
truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

print(input_data.shape, truth.shape)

# Encoded representation of shape (batch_size, encoding_size)
pfn_ae, pfn = get_pfn_ae([num_elements,element_size],phi_dim, [32,16])
#pfn, encoder = get_full_PFN([num_elements,element_size])

#(X_train, X_test,
# Y_train, Y_test) = train_test_split(input_data, truth, test_size=0.1)

X_eval, x_test, _, y_test = train_test_split(bkg, sig, test_size=0.1)
X_train, X_val, _, _ = train_test_split(X_eval, X_eval, test_size=0.2)


h = pfn_ae.fit(X_train, #Y_train,
        epochs=nepochs,
        batch_size=batchsize,
        validation_data= X_val,
        verbose=1)

phi_representation = pfn.predict(X_test)

print("Phi Space:")
#print(encoded)
#print(encoded.shape)
#print(type(encoded))
#print()
b = phi_representation.T
print(b)
print(b.shape)
c = b[[not (n==0).all() for n in b]]
print("Pruned:")
print(c)
print(c.shape)

## Plot loss
bkg_loss = []
sig_loss = []
nevents = len(y_test)
step_size = 4
for i in range(0,nevents, step_size):
  xt = x_test[i:i+step_size]
  yt = y_test[i:i+step_size]
  #xt = x[np.newaxis, :] #give x and y the correct shape (,64)
  #yt = y[np.newaxis, :]

  # NOTE - unclear why they are printed in this order, but it seems to be the case
  x_loss = pfn_ae.evaluate(xt, batch_size = step_size, verbose=0)
  y_loss = pfn_ae.evaluate(yt, batch_size = step_size, verbose=0)

  bkg_loss.append(x_loss)
  sig_loss.append(y_loss)
  if i%100 == 0: print("Processed", i, "events")

plot_loss(h,1)
plot_score(bkg_loss, sig_loss, False, "pfn_ae")

######## EVALUATE SUPERVISED ######
## get predictions on test data
#preds = pfn.predict(X_test, batch_size=1000)

## get ROC curve
#pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
#auc = roc_auc_score(Y_test[:,1], preds[:,1])
#print()
#print('PFN AUC:', auc)
#print()
