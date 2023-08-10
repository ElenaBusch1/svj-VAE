import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import json
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
batchsize_vae=32

pfn_model = 'PFNv3p1'
vae_model = 'vANTELOPE'
arch_dir = "architectures_saved/"
data_path = "/data/users/ebusch/SVJ/autoencoder/"

################### Train the VAE ###############################
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## AE events
x_events = 50000
y_events = 5000
#z_events = 199899

bkg_file = data_path + "v8.1/skim3.user.ebusch.QCDskim.root"
sig_file = data_path + "v8.1/skim3.user.ebusch.SIGskim.root"
bkg2 = getTwoJetSystem(x_events,bkg_file,[],use_weight=True)
sig2 = getTwoJetSystem(y_events,sig_file,[],use_weight=False)
#sig2 = getTwoJetSystem(y_events,bkg_file,[],use_weight=True)

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
#quit()

phi_evalb = (phi_evalb - eval_min)/(eval_max-eval_min)
phi_testb = (phi_testb - eval_min)/(eval_max-eval_min)
phi_sig = (phi_sig - eval_min)/(eval_max-eval_min)

#phi_evalb, phi_scaler = apply_StandardScaling(phi_evalb)
#phi_testb, _ = apply_StandardScaling(phi_testb,phi_scaler,False)
#phi_sig, _ = apply_StandardScaling(phi_sig,phi_scaler,False)

plot_phi(phi_evalb,"train","PFN_phi_train_scaled")
plot_phi(phi_testb,"test","PFN_phi_test_scaled")
plot_phi(phi_sig,"sig","PFN_phi_sig_scaled")

vae = get_vae(phi_dim,encoding_dim,latent_dim)

h2 = vae.fit(phi_evalb, 
    epochs=nepochs,
    batch_size=batchsize_vae,
    validation_split=0.2,
    verbose=1)

# # simple ae
# ae.save(arch_dir+ae_model)
# print("saved model")

#complex ae
vae.get_layer('encoder').save_weights(arch_dir+vae_model+'_encoder_weights.h5')
vae.get_layer('decoder').save_weights(arch_dir+vae_model+'_decoder_weights.h5')
vae.get_layer('encoder').save(arch_dir+vae_model+'_encoder_arch')
vae.get_layer('decoder').save(arch_dir+vae_model+'_decoder_arch')
#with open(arch_dir+vae_model+"8.1_history.json", "w") as f:
#    json.dump(h2.history, f)

######## EVALUATE SUPERVISED ######
# # --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h2, vae_model, 'loss')
#plot_loss(h2, vae_model, "kl_loss")
#plot_loss(h2, vae_model, "reco_loss")

#2. Get loss
#bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_sig)
"""
pred_phi_bkg = vae.predict(phi_testb)['reconstruction']
pred_phi_sig = vae.predict(phi_sig)['reconstruction']
bkg_loss = keras.losses.mse(phi_testb, pred_phi_bkg)
sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)
"""

bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig)

plot_score(bkg_loss, sig_loss, False, False, vae_model)
#plot_score(bkg_kl_loss, sig_kl_loss, False, False, vae_model+"_KLD")
#plot_score(bkg_reco_loss, sig_reco_loss, False, False, vae_model+"_Reco")

# # 3. Signal Sensitivity Score
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, vae_model, True)
#do_roc(bkg_reco_loss, sig_reco_loss, vae_model+"_Reco", True)
#do_roc(bkg_kl_loss, sig_kl_loss, vae_model+"_KLD", True)

print("Taking log of score...")
bkg_loss = np.log(bkg_loss)
sig_loss = np.log(sig_loss)
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, vae_model+'log', True)


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

