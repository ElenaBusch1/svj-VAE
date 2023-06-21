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
latent_dim = 12
phi_dim = 64
nepochs=50
batchsize_ae=32

pfn_model = 'PFNv1'
ae_model = 'ANTELOPE'
arch_dir = "architectures_saved/"
npy_dir = "npy_inputs/"

################### Train the AE ###############################
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## AE events
x_events = 200000
y_events = 20000

#bkg2, sig2 = getTwoJetSystem(x_events,y_events)
bkg2 = np.load(npy_dir+"bkg.npy")
sig2 = np.load(npy_dir+"sig.npy")

scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
sig2,_ = apply_StandardScaling(sig2,scaler,False)
plot_vectors(bkg2,sig2,"ANTELOPE")

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

#plot_score(phi_bkg[:,11], phi_sig[:,11], False, False, "phi_11_raw")

phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0])
plot_phi(phi_evalb,"Train","PFN_phi_train_raw")
plot_phi(phi_testb,"Test","PFN_phi_test_raw")
plot_phi(phi_sig,"Signal","PFN_phi_sig_raw")

eval_max = np.amax(phi_evalb)
eval_min = np.amin(phi_evalb)
sig_max = np.amax(phi_sig)
print("Min: ", eval_min)
print("Max: ", eval_max)
if (sig_max > eval_max): eval_max = sig_max
print("Final Max: ", eval_max)

phi_evalb = (phi_evalb - eval_min)/(eval_max-eval_min)
phi_testb = (phi_testb - eval_min)/(eval_max-eval_min)
phi_sig = (phi_sig - eval_min)/(eval_max-eval_min)

#phi_evalb, phi_scaler = apply_StandardScaling(phi_evalb)
#phi_testb, _ = apply_StandardScaling(phi_testb,phi_scaler,False)
#phi_sig, _ = apply_StandardScaling(phi_sig,phi_scaler,False)

plot_phi(phi_evalb,"train","PFN_phi_train_scaled")
plot_phi(phi_testb,"test","PFN_phi_test_scaled")
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

plot_score(bkg_loss, sig_loss, False, True, ae_model)

# # 3. Signal Sensitivity Score
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, ae_model, True)

print("Taking log of score...")
bkg_loss = np.log(bkg_loss)
sig_loss = np.log(sig_loss)
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, ae_model+'log', True)


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
