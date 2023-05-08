import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *

# Example usage
encoding_dim = 32
latent_dim = 4
phi_dim = 64
nepochs=30
batchsize_ae=32

pfn_model = 'PFN'
ae_model = 'PFN'
arch_dir = "architectures_saved/"

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

