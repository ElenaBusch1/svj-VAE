import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from models import *
from models_archive import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *

# Example usage
num_elements = 16
element_size = 4
encoding_size = 32
phi_dim = 128
nepochs=10
batchsize=100

model_name = 'PFN'
arch_dir = "architectures_saved/"
#encoder = PermInvEncoder(num_elements, element_size, encoding_size)

# Input of shape (batch_size, num_elements, element_size)
#input_data = np.random.randn(500, num_elements, element_size).astype(np.float32)
bkg_raw = read_vectors("../v6.4/v6p4smallQCD.root", 10000, False)
sig_raw = read_vectors("../v6.4/totalSig.root", 10000, False)

bkg2_raw = read_vectors("../v6.4/v6p4smallQCD.root", 10000, False)
sig2_raw = read_vectors("../v6.4/user.ebusch.515500.root", 10000, False)

bkg, sig = apply_EventScaling(bkg_raw, sig_raw)
bkg2, sig2 = apply_EventScaling(bkg2_raw, sig2_raw)

input_data = np.concatenate((bkg,sig),axis=0)

truth_bkg = np.zeros(bkg.shape[0])
truth_sig = np.ones(sig.shape[0])

truth_1D = np.concatenate((truth_bkg,truth_sig))
truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

print(input_data.shape, truth.shape)

# Encoded representation of shape (batch_size, encoding_size)
#pfn_ae, pfn = get_pfn_ae([num_elements,element_size],phi_dim, [32,16])
pfn,graph_orig = get_full_PFN([num_elements,element_size])

#(X_train, X_test,
# Y_train, Y_test) = train_test_split(input_data, truth, test_size=0.1)

x_eval, x_test, y_eval, y_test = train_test_split(input_data, truth, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(x_eval, y_eval, test_size=0.2)


h = pfn.fit(X_train, Y_train,
        epochs=nepochs,
        batch_size=batchsize,
        validation_data=(X_val,Y_val),
        verbose=1)

pfn.get_layer('graph').save_weights(arch_dir+model_name+'_graph_weights.h5')
pfn.get_layer('classifier').save_weights(arch_dir+model_name+'_classifier_weights.h5')
pfn.get_layer('graph').save(arch_dir+model_name+'_graph_arch')
pfn.get_layer('classifier').save(arch_dir+model_name+'_classifier_arch')

#phi_representation = pfn.predict(x_test)

#print("Phi Space:")
#print(encoded)
#print(encoded.shape)
#print(type(encoded))
#print()
#b = phi_representation.T
#print(b)
#print(b.shape)
#c = b[[not (n==0).all() for n in b]]
#print("Pruned:")
#print(c)
#print(c.shape)

graph = keras.models.load_model(arch_dir+'PFN_graph_arch')
graph.load_weights(arch_dir+'PFN_graph_weights.h5')
graph.compile()

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

phi_evalb, phi_testb, _, phi_tests = train_test_split(phi_bkg, phi_sig, test_size=0.1)

ae = get_ae(64,32,4)

h2 = ae.fit(phi_evalb, 
        epochs=nepochs,
        batch_size=batchsize,
        validation_split=0.2,
        verbose=1)

#bkg_pred = ae.predict(phi_testb)
#sig_pred = ae.predict(phi_tests)

#print(phi_testb)
#print(bkg_pred)
#print(sig_pred)

#bkg_loss = keras.losses.mse(bkg_pred, phi_testb)
#sig_loss = keras.losses.mse(sig_pred, phi_tests)
bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_tests)

print(bkg_loss)

#print(phi_rep)
#print(phi_rep.shape)

######## EVALUATE SUPERVISED ######
# # --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h2, 'trainPFN_AE', 'loss')
plot_score(bkg_loss, sig_loss, False, False, 'trainPFN_AE')

# ## get predictions on test data
# preds = pfn.predict(x_test)
# ## get ROC curve
# pfn_fp, pfn_tp, threshs = roc_curve(y_test[:,1], preds[:,1])
# #
# # get area under the ROC curve
# auc = roc_auc_score(y_test[:,1], preds[:,1])
# print()
# print('PFN AUC:', auc)
# print()
# 
# # plot distributions
# bkg_score = preds[:,1][y_test[:,1] == 0]
# sig_score = preds[:,1][y_test[:,1] == 1]
# plot_score(bkg_score, sig_score, False, False, 'PFN')
