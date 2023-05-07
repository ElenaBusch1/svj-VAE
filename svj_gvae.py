import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import *
from models_archive import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *

# Example usage
nevents = 50000
num_elements = 100
element_size = 4
encoding_dim = 32
latent_dim = 4
phi_dim = 64
nepochs=30
batchsize_pfn=500
batchsize_ae=32

pfn_model = 'PFN'
ae_model = 'PFN'
arch_dir = "architectures_saved/"

##  # Input of shape (batch_size, num_elements, element_size)
##  #input_data = np.random.randn(500, num_elements, element_size).astype(np.float32)
##  
##  bkg, sig = getTwoJetSystem(nevents,nevents)
##  
##  # 4. Plot inputs
##  #x_nz = remove_zero_padding(bkg)
##  #sig_nz = remove_zero_padding(sig)
##  plot_vectors(bkg,sig,"PFNrotated")
##  
##  #plot_nTracks(bkg_sel, sig_sel)
##  
##  # Create truth target
##  input_data = np.concatenate((bkg,sig),axis=0)
##  #input_data,_ = apply_StandardScaling(input_data)
##  
##  truth_bkg = np.zeros(bkg.shape[0])
##  truth_sig = np.ones(sig.shape[0])
##  
##  truth_1D = np.concatenate((truth_bkg,truth_sig))
##  truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)
##  
##  ## MAKE 1D DATA - DNN only
##  #input_data = input_data.reshape(input_data.shape[0],40*4)
##  
##  #plot_vectors(remove_zero_padding(bkg_scaled),remove_zero_padding(sig_scaled),"scaled")
##  
##  print("Training shape, truth shape")
##  print(input_data.shape, truth.shape)
##  # Encoded representation of shape (batch_size, encoding_size)
##  #pfn_ae, pfn = get_pfn_ae([num_elements,element_size],phi_dim, [32,16])
##  pfn,graph_orig = get_full_PFN([num_elements,element_size], phi_dim)
##  #pfn = get_dnn(160)
##  
##  #(X_train, X_test,
##  # Y_train, Y_test) = train_test_split(input_data, truth, test_size=0.1)
##  
##  x_train, x_test, y_train, y_test = train_test_split(input_data, truth, test_size=0.02)
##  #X_train, X_val, Y_train, Y_val = train_test_split(x_eval, y_eval, test_size=0.2)
##  
##  #x_train, x_test, y_train, y_test = train_test_split(x, x, test_size=sig.shape[0]) #done randomly
##  #x_valid, x_test, _, _ = train_test_split(x_temp, x_temp, test_size=0.5)
##  #x_train = reshape_3D(x_train, 16, 4)
##  #x_test = reshape_3D(x_test, 16, 4)
##  #sig = reshape_3D(sig, 16, 4)
##  x_train, scaler = apply_StandardScaling(x_train)
##  dump(scaler, arch_dir+pfn_model+'_scaler.bin', compress=True)
##  x_test,_ = apply_StandardScaling(x_test,scaler,False)
##  #sig,_ = apply_StandardScaling(sig,scaler,False)
##  
##  bkg_train_scaled = x_train[y_train[:,0] == 1]
##  sig_train_scaled = x_train[y_train[:,0] == 0]
##  bkg_test_scaled = x_test[y_test[:,0] == 1]
##  sig_test_scaled = x_test[y_test[:,0] == 0]
##  plot_vectors(bkg_train_scaled,sig_train_scaled,"PFNtrain")
##  plot_vectors(bkg_train_scaled,sig_train_scaled,"PFNtest")
##  
##  ## train
##  h = pfn.fit(x_train, y_train,
##      epochs=nepochs,
##      batch_size=batchsize_pfn,
##      validation_split=0.2,
##      verbose=1)
##  
##  ## save the model
##  pfn.get_layer('graph').save_weights(arch_dir+pfn_model+'_graph_weights.h5')
##  pfn.get_layer('classifier').save_weights(arch_dir+pfn_model+'_classifier_weights.h5')
##  pfn.get_layer('graph').save(arch_dir+pfn_model+'_graph_arch')
##  pfn.get_layer('classifier').save(arch_dir+pfn_model+'_classifier_arch')
##  
##  ## PFN training plots
##  # 1. Loss vs. epoch 
##  plot_loss(h, pfn_model, 'loss')
##  # 2. Score 
##  preds = pfn.predict(x_test)
##  bkg_score = preds[:,1][y_test[:,1] == 0]
##  sig_score = preds[:,1][y_test[:,1] == 1]
##  plot_score(bkg_score, sig_score, False, False, pfn_model)
##  n_test = min(len(sig_score),len(bkg_score))
##  bkg_score = bkg_score[:n_test]
##  sig_score = sig_score[:n_test]
##  do_roc(bkg_score, sig_score, "PFN", False)
##  
##  #quit()
################### Train the AE ###############################
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## AE events
x_events = 99570
y_events = 9957
bkg2, sig2 = getTwoJetSystem(x_events,y_events)
scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
sig2,_ = apply_StandardScaling(sig2,scaler,False)
plot_vectors(bkg2,sig2,"PFN_AE")

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

plot_phi(phi_bkg,"bkg","PFN_phi_bkg_raw")
plot_phi(phi_sig,"sig","PFN_phi_sig_raw")

phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0])

phi_evalb, phi_scaler = apply_StandardScaling(phi_evalb)
phi_testb, _ = apply_StandardScaling(phi_testb,phi_scaler,False)
phi_sig, _ = apply_StandardScaling(phi_sig,phi_scaler,False)

plot_phi(phi_bkg,"bkg","PFN_phi_bkg_scaled")
plot_phi(phi_sig,"sig","PFN_phi_sig_scaled")

ae = get_ae(phi_dim,encoding_dim,latent_dim)

h2 = ae.fit(phi_evalb, 
    epochs=nepochs,
    batch_size=batchsize_ae,
    validation_split=0.2,
    verbose=1)

# # simple ae
# ae.save(arch_dir+ae_model)
# print("saved model")

#complex ae
ae.get_layer('encoder').save_weights(arch_dir+ae_model+'_encoder_weights.h5')
ae.get_layer('decoder').save_weights(arch_dir+ae_model+'_decoder_weights.h5')
ae.get_layer('encoder').save(arch_dir+ae_model+'_encoder_arch')
ae.get_layer('decoder').save(arch_dir+ae_model+'_decoder_arch')

######## EVALUATE SUPERVISED ######
# # --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h2, ae_model, 'loss')

#2. Get loss
#bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_sig)
pred_phi_bkg = ae.predict(phi_testb)['reconstruction']
pred_phi_sig = ae.predict(phi_sig)['reconstruction']
bkg_loss = keras.losses.mse(phi_testb, pred_phi_bkg)
sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)

plot_score(bkg_loss, sig_loss, False, False, ae_model)

# # 3. Signal Sensitivity Score
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, ae_model, True)

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
