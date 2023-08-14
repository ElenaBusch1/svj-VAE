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
import time
# Example usage
encoding_dim = 32
latent_dim = 12
phi_dim = 64
#nepochs=20
nepochs=50
batchsize_vae=32

#pfn_model = 'PFNv1'
pfn_model = 'PFNv6'
vae_model = 'vANTELOPE' # vae change
#vae_model = 'ANTELOPE'
#arch_dir = "architectures_saved/"
arch_dir='/nevis/katya01/data/users/kpark/svj-vae/results/antelope/architectures_saved/'
#arch_dir = "/data/users/ebusch/SVJ/autoencoder/svj-vae/architectures_saved/"

################### Train the AE or VAE ###############################
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## AE events
sig_events = 50000
#sig_events = 20000
bkg_events = 100000
#bkg_events = 200000
#bkg2, sig2 = getTwoJetSystem(x_events,y_events)

track_array0 = ["jet0_GhostTrack_pt", "jet0_GhostTrack_eta", "jet0_GhostTrack_phi", "jet0_GhostTrack_e","jet0_GhostTrack_z0", "jet0_GhostTrack_d0", "jet0_GhostTrack_qOverP"]
track_array1 = ["jet1_GhostTrack_pt", "jet1_GhostTrack_eta", "jet1_GhostTrack_phi", "jet1_GhostTrack_e","jet1_GhostTrack_z0", "jet1_GhostTrack_d0", "jet1_GhostTrack_qOverP"]
jet_array = ["jet1_eta", "jet1_phi", "jet2_eta", "jet2_phi"] # order is important in apply_JetScalingRotation
bool_weight=True
bool_weight_sig=False
extraVars=['mT_jj', 'weight']
all_dir='/nevis/katya01/data/users/kpark/svj-vae/'
sig_file='skim3.user.ebusch.SIGskim.root'
bkg_file='skim1.user.ebusch.QCDskim.root'
#plot_dir=all_dir+'results/antelope/plots/'
plot_dir=all_dir+'results/antelope/plots_vae/' # vae change
seed=0
max_track=80
bool_pt=False
h5_dir=all_dir+'h5dir/antelope/aug10/'

sig, mT_sig, sig_sel, jet_sig, sig_in0, sig_in1 = getTwoJetSystem(nevents=sig_events,input_file=sig_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight_sig,  extraVars=extraVars, plot_dir=plot_dir, seed=seed,max_track=max_track, bool_pt=bool_pt,h5_dir=h5_dir, read_dir='/data/users/ebusch/SVJ/autoencoder/v8.1/')

bkg, mT_bkg, bkg_sel, jet_bkg, bkg_in0, bkg_in1 = getTwoJetSystem(nevents=bkg_events,input_file=bkg_file,
      track_array0=track_array0, track_array1=track_array1,  jet_array= jet_array,
      bool_weight=bool_weight,  extraVars=extraVars, plot_dir=plot_dir, seed=seed,max_track=max_track, bool_pt=bool_pt,h5_dir=h5_dir, read_dir='/data/users/ebusch/SVJ/autoencoder/v9.1/')

scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg,scaler,False) # change
sig2,_ = apply_StandardScaling(sig,scaler,False) # change
plot_vectors(bkg2,sig2,tag_file="ANTELOPE", tag_title=" (ANTELOPE)", plot_dir=plot_dir)# change

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

#plot_score(phi_bkg[:,11], phi_sig[:,11], False, False, "phi_11_raw")

phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0], random_state=42) # no info on labels e.g. y_train or y_test
#phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0])
plot_phi(phi_evalb,tag_file="PFN_phi_train_raw",tag_title="Train") # change
plot_phi(phi_testb,tag_file="PFN_phi_test_raw",tag_title="Test")
plot_phi(phi_sig,tag_file="PFN_phi_sig_raw", tag_title="Signal")

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

plot_phi(phi_evalb,tag_file="PFN_phi_train_scaled",tag_title="Train Scaled") # change
plot_phi(phi_testb,tag_file="PFN_phi_test_scaled",tag_title="Test Scaled")
plot_phi(phi_sig,tag_file="PFN_phi_sig_scaled", tag_title="Signal Scaled")



vae = get_vae(phi_dim,encoding_dim,latent_dim)

h2 = vae.fit(phi_evalb, 
    epochs=nepochs,
    batch_size=batchsize_vae,
    validation_split=0.2,
    verbose=1)

# # simple ae
#ae.save(arch_dir+vae_model)
#print("saved model"+ arch_dir+vae_model)


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
plot_loss(h2, loss='loss', tag_file=vae_model, tag_title=vae_model, plot_dir=plot_dir)

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
#start = time.time()
bkg_loss, sig_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(vae, phi_testb, phi_sig)
#end = time.time()
print("Elapsed (with get_multi_loss) = %s" % (end - start))
try:cprint(f'{min(bkg_loss)}, {min(sig_loss)}, {max(bkg_loss)}, {max(sig_loss)}', 'yellow')
except: cprint(f'{np.min(bkg_loss)}, {np.min(sig_loss)},{np.max(bkg_loss)}, {np.max(sig_loss)}', 'blue')
print('\n')
plot_score(bkg_loss, sig_loss, False, True, tag_file=vae_model, tag_title=vae_model, plot_dir=plot_dir, bool_pfn=False) # anomaly score
plot_score(bkg_loss, sig_loss, False, True, tag_file=vae_model+"_KLD", tag_title=vae_model+"_KLD", plot_dir=plot_dir, bool_pfn=False) # anomaly score
plot_score(bkg_loss, sig_loss, False, True, tag_file=vae_model+"_Reco", tag_title=vae_model+"_Reco", plot_dir=plot_dir, bool_pfn=False) # anomaly score
#plot_score(bkg_loss, sig_loss, False, True, vae_model)
#plot_score(bkg_kl_loss, sig_kl_loss, False, False, vae_model+"_KLD")
#plot_score(bkg_reco_loss, sig_reco_loss, False, False, vae_model+"_Reco")

"""
"""
# # 3. Signal Sensitivity Score
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, tag_file=vae_model, tag_title=vae_model,make_transformed_plot= False, plot_dir=plot_dir, bool_pfn=False)

#do_roc(bkg_reco_loss, sig_reco_loss, vvae_model+"_Reco", True)
#do_roc(bkg_kl_loss, sig_kl_loss, vvae_model+"_KLD", True)

print("Taking log of score...")
bkg_loss = np.log(bkg_loss)
sig_loss = np.log(sig_loss)
score = getSignalSensitivityScore(bkg_loss, sig_loss)
print("95 percentile score = ",score)
# # 4. ROCs/AUCs using sklearn functions imported above  
do_roc(bkg_loss, sig_loss, tag_file=vae_model+'log', tag_title=vae_model+'log',make_transformed_plot= True, plot_dir=plot_dir,  bool_pfn=False)

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

